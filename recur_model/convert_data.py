"""Convert FineWeb-Edu Parquet shards for LLaMA-Factory pretraining."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SCHEMA = pa.schema([("text", pa.string())])


def convert_shard(source: Path, destination: Path) -> None:
    with pq.ParquetFile(source) as parquet:
        schema = parquet.schema_arrow
        index = schema.get_field_index("text")
        if index == -1:
            raise ValueError(f"{source}: expected one 'text' column")

        text_type = schema.field(index).type
        if not (pa.types.is_string(text_type) or pa.types.is_large_string(text_type)):
            raise ValueError(f"{source}: 'text' must be a string column, got {text_type}")

        with pq.ParquetWriter(destination, TEXT_SCHEMA, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
                text = batch.column(0)
                if text.null_count:
                    raise ValueError(f"{source}: 'text' contains null values")

                writer.write_batch(pa.record_batch([text.cast(pa.string())], schema=TEXT_SCHEMA))


def convert_data(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}. Specify a new --output-dir.")

    shards = {split: sorted((source_dir / split).glob("*.parquet")) for split in ("train", "valid")}
    if not shards["train"]:
        raise FileNotFoundError(f"No training Parquet shards found in {source_dir / 'train'}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as directory:
        bundle = Path(directory) / "dataset"
        dataset_info = {}
        for split, sources in shards.items():
            if not sources:
                continue

            split_dir = bundle / split
            split_dir.mkdir(parents=True)
            for source in sources:
                convert_shard(source, split_dir / source.name)

            dataset_info[f"fineweb_edu_{split}"] = {
                "file_name": split,
                "columns": {"prompt": "text"},
            }

        (bundle / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
        bundle.rename(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifact" / "data" / "fineweb-edu")
    args = parser.parse_args()
    try:
        destination = convert_data(args.source_dir, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(destination)


if __name__ == "__main__":
    main()
