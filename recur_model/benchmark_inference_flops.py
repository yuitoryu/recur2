"""Count inference FLOPs at the model's maximum sequence length.

Counts cover matrix multiplications and depth scoring and mixing, with two
FLOPs per multiply-accumulate. They exclude normalization, softmax, biases,
and other elementwise operations.
"""

import argparse
import csv
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

from .configuration_recur import RecurConfig
from .modeling_recur import AttentionResidual, RecurForCausalLM


def native_self_attention_flops(query_shape, *args, **kwargs):
    """Count the four projections and two dense matmuls inside native MHA."""
    batch, length, width = query_shape
    projections = 8 * batch * length * width**2
    score_and_value_matmuls = 4 * batch * length**2 * width
    return projections + score_and_value_matmuls


def measure_flops(config, batch_size, device):
    model = RecurForCausalLM(config).to(device=device, dtype=torch.float32).eval()
    input_ids = torch.randint(
        config.vocab_size, (batch_size, config.max_position_embeddings), device=device
    )

    depth_mix_flops = 0

    def count_depth_mix(_module, args):
        nonlocal depth_mix_flops
        (sources,) = args
        # A dot product per source and a weighted sum across sources.
        depth_mix_flops += 4 * len(sources) * sources[0].numel()

    for module in model.modules():
        if isinstance(module, AttentionResidual):
            module.register_forward_pre_hook(count_depth_mix)

    # The default counter covers linear layers and SDPA. Native MHA is a single
    # operator that also includes the input and output projections.
    counter = FlopCounterMode(
        display=False,
        custom_mapping={torch.ops.aten._native_multi_head_attention: native_self_attention_flops},
    )
    with torch.inference_mode(), counter:
        model(input_ids=input_ids, use_cache=False)

    return counter.get_total_flops(), depth_mix_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--pattern", nargs=2, type=int, action="append", metavar=("LAYERS", "REPEATS"),
        help="Repeatable. Defaults to 10 6 and 16 1.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([
        "layers", "repeats", "batch_size", "seq_len",
        "matmul_gflops", "depth_mix_gflops", "total_gflops",
    ])
    for layers, repeats in args.pattern or [(10, 6), (16, 1)]:
        config = RecurConfig(
            variant="scaled_block_attnres",
            hidden_size=640,
            num_attention_heads=8,
            intermediate_size=2560,
            activation="relu",
            max_position_embeddings=2048,
            vocab_size=50257,
            recur_pattern=[[layers, repeats]],
            block_size=4,
            residual_scale_base=1.0,
            residual_depth_base=10,
        )
        matmul_flops, depth_mix_flops = measure_flops(config, args.batch_size, args.device)
        writer.writerow([
            layers, repeats, args.batch_size, config.max_position_embeddings,
            f"{matmul_flops / 1e9:.3f}", f"{depth_mix_flops / 1e9:.3f}",
            f"{(matmul_flops + depth_mix_flops) / 1e9:.3f}",
        ])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
