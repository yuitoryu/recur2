"""Validate model-dependent arguments, then run LLaMA-Factory training."""

from pathlib import Path

from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.parser import read_args
from llamafactory.train.tuner import run_exp
from transformers.utils import is_flash_attn_2_available

from .configuration_recur import RecurConfig


def train(args):
    if args.get("flash_attn") == "fa2" and not is_flash_attn_2_available():
        raise ValueError("flash_attn: fa2 requires the CUDA flash-attn package. Run scripts/setup-wsl.sh --fa2.")
    model_dir = Path(args["model_name_or_path"])
    config = RecurConfig.from_pretrained(model_dir, local_files_only=True)
    config.validate_cutoff(args.get("cutoff_len", DataArguments.cutoff_len))
    run_exp(args)


def main():
    args = read_args()
    if not isinstance(args, dict):
        raise SystemExit("Usage: python -m recur_model.train TRAIN.yaml [key=value ...]")
    train(args)


if __name__ == "__main__":
    main()
