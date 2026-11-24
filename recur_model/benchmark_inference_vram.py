"""Measure peak CUDA memory for inference at the model's maximum sequence length."""

import argparse
import csv
import sys

import torch

from .configuration_recur import RecurConfig
from .modeling_recur import RecurForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32", help="Model dtype (default: fp32)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    config = RecurConfig(
        variant="scaled_block_attnres",
        hidden_size=640,
        num_attention_heads=8,
        intermediate_size=2560,
        activation="relu",
        max_position_embeddings=2048,
        vocab_size=50257,
        recur_pattern=[[10, 6]],
        block_size=4,
        residual_scale_base=1.0,
        residual_depth_base=10,
    )
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    model = RecurForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    if args.compile:
        # Use a static graph and report any graph breaks as errors.
        model = torch.compile(model, dynamic=False, fullgraph=True)

    length = config.max_position_embeddings
    input_ids = torch.randint(config.vocab_size, (args.batch_size, length), device="cuda")
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()

        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del output

    mib = 2**20
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow([
        "batch_size", "seq_len", "compiled", "dtype", "baseline_mib",
        "peak_allocated_mib", "peak_above_baseline_mib",
    ])
    writer.writerow([
        args.batch_size, length, args.compile, args.dtype, f"{baseline / mib:.3f}",
        f"{peak / mib:.3f}", f"{(peak - baseline) / mib:.3f}",
    ])


if __name__ == "__main__":
    main()
