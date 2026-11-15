"""Run from the repository root in the WSL FA2 environment.

    .venv/bin/python -m recur_model.benchmark_block_liger \
        --shard ../recurtransformer/data/train/000_00000.parquet
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter

import torch
from datasets import Dataset, load_from_disk
from llamafactory.data.processor.pretrain import PretrainDatasetProcessor
from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments, TrainingArguments
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.pt.trainer import CustomTrainer
from pyarrow.parquet import ParquetFile
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainerCallback, set_seed

from .configuration_recur import RecurConfig
from .modeling_recur import RecurForCausalLM
from .prepare import prepare_model


SEQUENCE = 512
SEED = 42
WARMUP = 5
PROFILE_STEP = WARMUP + 1
TRIALS = 20
STEPS = PROFILE_STEP + TRIALS
ORDER = ("baseline", "liger", "liger", "baseline")
PINNED = {
    "torch": "2.8.0+cu128", "triton": "3.4.0", "transformers": "4.57.6",
    "flash-attn": "2.8.3", "llamafactory": "0.9.5", "liger-kernel": "0.6.2",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_experiment(root, shard, tokenizer_source):
    shape = {
        "use_model": "RecurrentTransformerBlockAttnRes", "d_model": 320, "nhead": 5,
        "dim_feedforward": 1248, "max_seq_len": 2048, "vocab_size": 50257,
        "recur_pattern": [[2, 1], [2, 4], [2, 1]], "block_size": 4,
    }
    write_json(root / "model.json", shape)
    baseline = root / "baseline"
    prepare_model(root / "model.json", baseline, tokenizer_source=tokenizer_source, cutoff_len=SEQUENCE)
    set_seed(SEED)
    RecurForCausalLM(RecurConfig.from_pretrained(baseline)).save_pretrained(baseline)

    candidate = root / "liger"
    shutil.copytree(baseline, candidate)
    shutil.copyfile(Path(__file__).with_name("modeling_block_liger.py"), candidate / "modeling_block_liger.py")
    config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
    config["architectures"] = ["BlockAttnResLigerForCausalLM"]
    config["auto_map"]["AutoModelForCausalLM"] = "modeling_block_liger.BlockAttnResLigerForCausalLM"
    write_json(candidate / "config.json", config)

    # Project FineWeb-Edu shards are Parquet. Read the first 256 documents of just one shard.
    parquet = ParquetFile(shard)
    rows = next(parquet.iter_batches(batch_size=256, columns=["text"]), None)
    if rows is None:
        raise ValueError(f"No training documents in {shard}")
    texts = rows.column(0).to_pylist()
    if any(not isinstance(text, str) for text in texts):
        raise ValueError(f"Expected non-null text in {shard}")
    tokenizer = AutoTokenizer.from_pretrained(baseline, local_files_only=True)
    processor = PretrainDatasetProcessor(None, tokenizer, None, DataArguments(packing=True, cutoff_len=SEQUENCE))
    packed = processor.preprocess_dataset({"_prompt": [[{"content": text}] for text in texts]})
    if len(packed["input_ids"]) < STEPS:
        raise ValueError(f"Need at least {STEPS} packed batches from {shard}")
    batches = Dataset.from_dict(packed).select(range(STEPS))
    batches.save_to_disk(root / "packed")
    return shape, batches._fingerprint


def verify_parity(root):
    models = [
        AutoModelForCausalLM.from_pretrained(
            root / arm, local_files_only=True, trust_remote_code=True, attn_implementation="flash_attention_2"
        ).cuda()
        for arm in ORDER[:2]
    ]
    baseline, candidate = models
    for model in models:
        # Disable dropout for numerical parity, including internal attention dropout.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            elif isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = 0.0
    torch.testing.assert_close(baseline.state_dict(), candidate.state_dict(), rtol=0, atol=0)
    batch = load_from_disk(root / "packed").with_format("torch")[0]
    ids = batch["input_ids"][:32].unsqueeze(0).cuda()
    mask = torch.ones_like(ids)
    padded = mask.clone()
    padded[:, :5] = 0
    ignored = ids.clone()
    ignored[:, 10:15] = -100
    all_ignored = torch.full_like(ids, -100)
    cases = {
        "dense": (ids, mask, ids),
        "padded_and_ignored": (ids, padded, ignored),
        "all_ignored": (ids, mask, all_ignored),
        "length_one": (ids[:, :1], mask[:, :1], ids[:, :1]),
    }
    results = {}
    for case, (inputs, attention_mask, labels) in cases.items():
        losses = []
        for model in models:
            model.train().zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            losses.append(loss.detach())
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-3, atol=2e-3)
        max_relative_error = 0.0
        for (name, ref), (_, test) in zip(baseline.named_parameters(), candidate.named_parameters(), strict=True):
            if ref.grad is None or test.grad is None:
                if ref.grad is not None or test.grad is not None:
                    raise AssertionError(f"{case}: gradient presence differs for {name}")
                continue
            scale = ref.grad.float().norm().item()
            difference = (ref.grad.float() - test.grad.float()).norm().item()
            # Chunked BF16 matmuls change rounding. Compare each parameter's gradient as a vector.
            tolerance = 0.03 * scale + 1e-5
            if not math.isfinite(difference) or difference > tolerance:
                raise AssertionError(f"{case}: {name} gradient L2 error {difference:g}, reference norm {scale:g}")
            max_relative_error = max(max_relative_error, difference / max(scale, 1e-6))
        results[case] = {"baseline_loss": losses[0].item(), "liger_loss": losses[1].item(),
                         "max_parameter_gradient_relative_l2": max_relative_error}

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference, test = [model.eval()(ids, attention_mask=padded, labels=ignored) for model in models]
        torch.testing.assert_close(reference.logits, test.logits, rtol=0, atol=0)
        torch.testing.assert_close(reference.loss, test.loss, rtol=0, atol=0)
    write_json(root / "parity.json", results)
    print("Loss, gradients, weight compatibility, and evaluation logits passed parity.", flush=True)


class MeasureSteps(TrainerCallback):
    def __init__(self, output, arm):
        self.output = output
        self.arm = arm
        self.warmup_seconds = []
        self.samples = []
        self.kernels = {}

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if state.global_step == WARMUP:
            self.trace = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.trace.start()
        self.start = perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        seconds = perf_counter() - self.start
        if state.global_step <= WARMUP:
            self.warmup_seconds.append(seconds)
        elif state.global_step == PROFILE_STEP:
            self.trace.stop()
            events = {event.name for event in self.trace.events()}
            self.kernels = {
                "fa2_forward": sorted(n for n in events if n == "flash_attn::_flash_attn_forward"),
                "fa2_backward": sorted(n for n in events if n == "flash_attn::_flash_attn_backward"),
                "compiled_forward": sorted(n for n in events if "Torch-Compiled Region" in n),
                "compiled_backward": sorted(n for n in events if n == "CompiledFunctionBackward"),
                "liger_cross_entropy": sorted(n for n in events if "liger_cross_entropy_kernel" in n),
            }
            write_json(self.output / "kernels.json", self.kernels)
            required = ("fa2_forward", "fa2_backward", "compiled_forward", "compiled_backward")
            if not all(self.kernels[key] for key in required):
                self.trace.export_chrome_trace(str(self.output / "failed-trace.json"))
                raise RuntimeError(f"{self.arm} profile did not confirm FA2 and compiled execution: {self.kernels}")
            if bool(self.kernels["liger_cross_entropy"]) != (self.arm == "liger"):
                raise RuntimeError(f"Unexpected Liger kernel activity for {self.arm}: {self.kernels}")
        else:
            self.samples.append({"seconds": seconds, "peak_bytes": torch.cuda.max_memory_allocated()})


def train_arm(root, arm, number):
    output = root / f"{number:02d}-{arm}"
    output.mkdir()
    args = TrainingArguments(
        output_dir=str(output), do_train=True, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        max_steps=STEPS, optim="adamw_torch", learning_rate=5e-4, bf16=True,
        torch_compile=True, torch_compile_backend="inductor", torch_compile_mode="default",
        save_strategy="no", logging_strategy="no", report_to="none", disable_tqdm=True,
        seed=SEED, data_seed=SEED, logging_nan_inf_filter=False,
    )
    os.environ["ACCELERATE_DYNAMO_FULLGRAPH"] = "false"
    model_args = ModelArguments(
        model_name_or_path=str(root / arm), trust_remote_code=True, flash_attn="fa2",
        disable_gradient_checkpointing=True,
    )
    model_args.compute_dtype = torch.bfloat16
    finetuning = FinetuningArguments(stage="pt", finetuning_type="full", disable_shuffling=True)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning, is_trainable=True)
    if model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("LLaMA-Factory failed to select FlashAttention-2")
    timer = MeasureSteps(output, arm)
    trainer = CustomTrainer(
        model=model, args=args, finetuning_args=finetuning, callbacks=[timer],
        train_dataset=load_from_disk(root / "packed"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        **tokenizer_module,
    )
    result = trainer.train()
    if len(timer.samples) != TRIALS or not math.isfinite(result.training_loss):
        raise RuntimeError(f"{arm} did not produce {TRIALS} measured steps with finite training loss")
    median_seconds = median(s["seconds"] for s in timer.samples)
    report = {
        "arm": arm, "median_step_ms": median_seconds * 1000, "tokens_per_second": SEQUENCE / median_seconds,
        "peak_mib": max(s["peak_bytes"] for s in timer.samples) / 2**20,
        "warmup_seconds": timer.warmup_seconds, "warmup_total_seconds": sum(timer.warmup_seconds),
        "training_loss": result.training_loss, "samples": timer.samples, "kernels": timer.kernels,
    }
    write_json(output / "metrics.json", report)
    print(f"{arm}: {report['tokens_per_second']:.1f} tokens/s, {report['peak_mib']:.1f} MiB", flush=True)


def run_worker(root, task, number=0):
    command = [sys.executable, "-m", __spec__.name, "--worker", task, "--root", str(root), "--run", str(number)]
    # Isolate cold compilation and keep generated files on the WSL filesystem.
    with TemporaryDirectory(prefix="recur-compile-") as directory:
        cache = Path(directory)
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
        subprocess.run(command, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, help="Local FineWeb-Edu training Parquet shard")
    parser.add_argument("--tokenizer", default="artifacts/attnres-50m", help="Project GPT-2 tokenizer path or Hub ID")
    parser.add_argument("--resume", type=Path, help="Resume this benchmark's unfinished runs from a results directory")
    parser.add_argument("--worker", choices=("parity", "baseline", "liger"), help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == "parity":
        verify_parity(args.root)
        return
    if args.worker:
        train_arm(args.root, args.worker, args.run)
        return
    if (args.shard is None) == (args.resume is None):
        parser.error("provide exactly one of --shard and --resume")
    if args.shard is not None and not args.shard.is_file():
        parser.error("--shard must point to a local FineWeb-Edu training Parquet shard")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA with native BF16 support is required")
    versions = {package: version(package) for package in PINNED}
    if versions != PINNED:
        parser.error(f"Expected pinned WSL packages {PINNED}, found {versions}")

    gpu = torch.cuda.get_device_name()
    if args.resume is None:
        Path("runs").mkdir(exist_ok=True)
        root = Path(mkdtemp(prefix="block-liger.", dir="runs")).resolve()
        print(f"Benchmark outputs: {root}", flush=True)
        shape, fingerprint = prepare_experiment(root, args.shard, args.tokenizer)
        metadata = {
            "gpu": gpu, "versions": versions, "model": shape, "sequence": SEQUENCE,
            "batch_size": 1, "shard": str(args.shard.resolve()), "packed_fingerprint": fingerprint, "order": ORDER,
            "optimizer": "adamw_torch", "learning_rate": 5e-4, "seed": SEED, "attention": "flash_attention_2",
            "precision": "bf16", "compiler": "inductor", "fullgraph": False,
        }
        write_json(root / "setup.json", metadata)
    else:
        root = args.resume.resolve()
        metadata = json.loads((root / "setup.json").read_text(encoding="utf-8"))
        if metadata["gpu"] != gpu or metadata["versions"] != versions or tuple(metadata["order"]) != ORDER:
            parser.error("--resume requires the same GPU, package versions, and experiment order")
        print(f"Resuming benchmark: {root}", flush=True)
    if not (root / "parity.json").exists():
        run_worker(root, "parity")
    runs = []
    for number, arm in enumerate(ORDER, 1):
        output = root / f"{number:02d}-{arm}"
        metrics = output / "metrics.json"
        if not metrics.exists():
            if output.exists():
                archive = Path(mkdtemp(prefix=f"{output.name}-failed.", dir=root))
                output.rename(archive / output.name)
            run_worker(root, arm, number)
        runs.append(json.loads(metrics.read_text(encoding="utf-8")))
    summary = {}
    for arm in ORDER[:2]:
        selected = [r for r in runs if r["arm"] == arm]
        seconds = median(s["seconds"] for r in selected for s in r["samples"])
        summary[arm] = {"median_step_ms": seconds * 1000, "tokens_per_second": SEQUENCE / seconds,
                        "peak_mib": max(r["peak_mib"] for r in selected)}
    speed_change = 100 * (summary["liger"]["tokens_per_second"] / summary["baseline"]["tokens_per_second"] - 1)
    write_json(root / "results.json", {**metadata, "summary": summary, "speed_change_percent": speed_change, "runs": runs})
    print(f"Liger throughput change: {speed_change:+.1f}% ({root / 'results.json'})", flush=True)


if __name__ == "__main__":
    main()
