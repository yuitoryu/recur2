import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer

from .configuration_recur import RecurConfig


def prepare_model(source, destination, *, tokenizer_source, cutoff_len=None):
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Model directory already exists: {destination}")

    config = RecurConfig.from_legacy(json.loads(source.read_text(encoding="utf-8-sig")))
    if cutoff_len is not None:
        config.validate_cutoff(cutoff_len)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if len(tokenizer) != config.vocab_size or tokenizer.eos_token_id is None:
        raise ValueError(
            f"Expected a tokenizer with {config.vocab_size} tokens and an EOS token; got {len(tokenizer)} tokens"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = config.max_position_embeddings
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.architectures = ["RecurForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_recur.RecurConfig",
        "AutoModelForCausalLM": "modeling_recur.RecurForCausalLM",
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        bundle = Path(directory) / "model"
        config.save_pretrained(bundle)
        tokenizer.save_pretrained(bundle)
        for filename in ("configuration_recur.py", "modeling_recur.py"):
            shutil.copyfile(Path(__file__).with_name(filename), bundle / filename)
        bundle.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description="Prepare a local Hugging Face directory from model.json.")
    parser.add_argument("model_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer", default="openai-community/gpt2", help="GPT-2 Hub ID or local tokenizer directory")
    parser.add_argument("--cutoff-len", type=int)
    args = parser.parse_args()
    print(prepare_model(args.model_json, args.output_dir, tokenizer_source=args.tokenizer, cutoff_len=args.cutoff_len))


if __name__ == "__main__":
    main()
