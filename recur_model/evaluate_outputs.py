"""Measure layer and loop output changes on the pretraining validation set."""

import argparse
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.misc import has_tokenized_data
from llamafactory.hparams import DataArguments, ModelArguments
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, HfArgumentParser, Seq2SeqTrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

from .configuration_recur import RecurConfig
from .modeling_recur import RecurForCausalLM


BLOCK_VARIANTS = ("block_attnres", "scaled_block_attnres")


def validate_evaluation_config(config):
    if config.variant in ("plain", "scaled_plain"):
        return
    if config.variant not in BLOCK_VARIANTS:
        raise ValueError(f"Unsupported variant for output evaluation: {config.variant}")
    if len(config.recur_pattern) != 1:
        raise ValueError(f"Expected one group [[layers, repeats]] looping over all layers, got {config.recur_pattern}")
    if config.block_size % 2:
        raise ValueError(f"Block AttnRes evaluation requires an even block_size, got {config.block_size}")
    layers, _ = config.recur_pattern[0]
    layers_per_block = config.block_size // 2
    if layers % layers_per_block:
        raise ValueError(f"{layers} layers per loop must be divisible by block_size / 2 ({layers_per_block})")


def compile_layers(model):
    if model.config.variant in BLOCK_VARIANTS:
        # Compile shared functions so each physical layer can reuse the same graphs.
        attention_forward = torch.compile(torch.nn.MultiheadAttention.forward)
        linear_forward = torch.compile(torch.nn.Linear.forward)
        for group in model.layers:
            for layer in group:
                layer.self_attn.forward = partial(attention_forward, layer.self_attn)
                for linear in (layer.linear1, layer.linear2):
                    linear.forward = partial(linear_forward, linear)
        return
    for group in model.layers:
        for layer in group.layers:
            layer.forward = torch.compile(layer.forward)


def validation_data(settings, checkpoint, tokenizer, packing):
    settings = dict(settings, model_name_or_path=str(checkpoint))
    training_packing = settings.get("packing")
    # Factory applies its implicit PT packing default after DataArguments initialization.
    # Explicit packing in the training YAML also affects the effective cutoff length.
    if not packing:
        settings["packing"] = False
    model_args, data_args = HfArgumentParser((ModelArguments, DataArguments)).parse_dict(settings, allow_extra_keys=True)
    data_args.packing = packing
    if data_args.eval_dataset:
        data_args.dataset = None
    if data_args.tokenized_path and (not has_tokenized_data(data_args.tokenized_path)
                                     or packing != (training_packing is not False)):
        data_args.tokenized_path = None
    seed = settings.get("seed", 42)
    set_seed(seed)
    training_args = Seq2SeqTrainingArguments(output_dir=str(checkpoint), seed=seed, report_to="none",
                                            dataloader_num_workers=max(1, settings.get("dataloader_num_workers", 0)))
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    datasets = get_dataset(template, model_args, data_args, training_args, "pt", tokenizer)
    if "eval_dataset" not in datasets:
        raise ValueError("Training YAML must specify a validation dataset (eval_dataset or val_size).")
    validation = datasets["eval_dataset"]
    return validation if isinstance(validation, dict) else {"validation": validation}


def output_metrics(earlier, later, mask):
    """Return distance and cosine rows, each ending with its whole-sample value."""
    earlier = earlier.float()
    later = later.float()
    delta = (later - earlier).norm(dim=-1).masked_fill(~mask, 0)
    base = earlier.norm(dim=-1).masked_fill(~mask, 0)
    base_whole = base.norm(dim=-1, keepdim=True)
    whole = delta.norm(dim=-1, keepdim=True) / (base_whole + 1e-8)
    distance = torch.cat((delta / (base + 1e-8), whole), dim=-1)

    dot = (earlier * later).sum(dim=-1).masked_fill(~mask, 0)
    later_norm = later.norm(dim=-1).masked_fill(~mask, 0)
    numerator = torch.cat((dot, dot.sum(dim=-1, keepdim=True)), dim=-1)
    denominator = torch.cat((base * later_norm, base_whole * later_norm.norm(dim=-1, keepdim=True)), dim=-1)
    cosine = torch.where(denominator == 0, 0.0, numerator / denominator)
    return torch.stack((distance, cosine), dim=1)


@torch.inference_mode()
def collect_batch(model, batch, *, measure_layers=True, measure_loops=True):
    mask = batch["attention_mask"].bool()
    positions = iter(
        (group, repeat, layer, depth)
        for group, (depth, repeats) in enumerate(model.config.recur_pattern, 1)
        for repeat in range(1, repeats + 1)
        for layer in range(1, depth + 1)
        if measure_layers or layer == depth
    )
    comparisons = []
    previous_layer = first_loop = previous_loop = None
    block = model.config.variant in BLOCK_VARIANTS
    pending_loop = None

    def record_loop(group, repeat, output):
        nonlocal first_loop, previous_loop
        if repeat == 1:
            first_loop = output
        else:
            adjacent = output_metrics(previous_loop, output, mask)
            position = dict(kind="adjacent_loops", group=group, from_loop=repeat - 1, to_loop=repeat)
            comparisons.append((position, adjacent))
            from_first = adjacent if repeat == 2 else output_metrics(first_loop, output, mask)
            comparisons.append((dict(kind="from_first_loop", group=group, from_loop=1, to_loop=repeat), from_first))
        previous_loop = output

    def capture_boundary(_module, inputs):
        nonlocal pending_loop
        if pending_loop is not None:
            record_loop(*pending_loop, inputs[0])
            pending_loop = None

    def record_layer(_module, _inputs, output, *, scale=1.0):
        nonlocal previous_layer, pending_loop
        group, repeat, layer, depth = next(positions)
        if measure_layers:
            if scale != 1.0:
                output = scale * output
            if layer > 1:
                position = dict(kind="adjacent_layers", group=group, loop=repeat, from_layer=layer - 1, to_layer=layer)
                comparisons.append((position, output_metrics(previous_layer, output, mask)))
            previous_layer = output
        if measure_loops and layer == depth:
            if block:
                pending_loop = (group, repeat)
            else:
                record_loop(group, repeat, output)

    with ExitStack() as hooks:
        for group in model.layers:
            group_layers = group if block else group.layers
            if block and measure_loops:
                hooks.enter_context(group_layers[0].norm1.register_forward_pre_hook(capture_boundary))
            capture_layers = group_layers if measure_layers else (group_layers[-1],)
            for layer in capture_layers:
                if block:
                    hook = partial(record_layer, scale=layer.residual_scale)
                    hooks.enter_context(layer.dropout2.register_forward_hook(hook))
                else:
                    hooks.enter_context(layer.register_forward_hook(record_layer))
        if block and measure_loops:
            hooks.enter_context(model.final_norm.register_forward_pre_hook(capture_boundary))
        model(**batch, use_cache=False)

    visible = mask.cpu()
    rows = [dict(valid_length=int(m.sum()), comparisons=[]) for m in visible]
    if not comparisons:
        return rows
    locations, measurements = zip(*comparisons)
    for position, values in zip(locations, torch.stack(measurements).cpu()):
        for row, m, (distance, cosine) in zip(rows, visible, values):
            row["comparisons"].append(dict(**position, tokenwise=distance[:-1][m].tolist(), whole=distance[-1].item(),
                                           tokenwise_cosine=cosine[:-1][m].tolist(), whole_cosine=cosine[-1].item()))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", type=Path, help="Training configuration directory containing train.yaml")
    parser.add_argument("run_dir", type=Path, help="LLaMA-Factory output directory containing checkpoint-<step>")
    parser.add_argument("--train-yaml", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Compile layer computations outside the capture hooks")
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument("--layers", action="store_true", help="Compare adjacent layers")
    parser.add_argument("--loops", action="store_true", help="Compare loop outputs with the previous and first loops")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not (args.layers or args.loops):
        parser.error("Enable at least one of --layers and --loops")
    latest = get_last_checkpoint(str(args.run_dir))
    if latest is None:
        parser.error(f"No checkpoint-<step> found in {args.run_dir}")
    checkpoint = Path(latest)
    config = RecurConfig.from_pretrained(checkpoint)
    try:
        validate_evaluation_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    settings = OmegaConf.to_container(OmegaConf.load(args.train_yaml or args.config_dir / "train.yaml"), resolve=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=settings.get("use_fast_tokenizer", True))
    datasets = validation_data(settings, checkpoint, tokenizer, not args.no_packing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecurForCausalLM.from_pretrained(checkpoint, config=config, dtype="auto").to(device).eval()
    if args.compile:
        compile_layers(model)

    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    step = checkpoint.name.split("-")[-1]
    suffix = "" if args.layers and args.loops else ("-layers" if args.layers else "-loops")
    destination = output_dir / f"output_differences-{step}{suffix}.jsonl"
    collate = DataCollatorWithPadding(tokenizer)
    count = 0
    print(f"Evaluating {checkpoint} on {device}", flush=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        partial = Path(temp) / destination.name
        with partial.open("w", encoding="utf-8") as stream:
            for name, dataset in datasets.items():
                loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
                for batch in tqdm(loader, desc=f"Evaluating {name}", unit="batch", miniters=1):
                    inputs = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
                    for row in collect_batch(model, inputs, measure_layers=args.layers, measure_loops=args.loops):
                        stream.write(json.dumps(dict(sample_id=count, dataset=name, **row), separators=(",", ":")) + "\n")
                        count += 1
        partial.replace(destination)
    print(f"Wrote {count} samples to {destination}")


if __name__ == "__main__":
    main()
