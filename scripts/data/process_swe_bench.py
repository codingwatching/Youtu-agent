"""Import SWE-bench dataset into the evaluation database.

Usage:
    python scripts/data/process_swe_bench.py --subset verified --split test
"""

import argparse
import json

import datasets
from sqlmodel import Session, select

from utu.db.eval_datapoint import DatasetSample
from utu.utils import SQLModelUtils

engine = SQLModelUtils.get_engine()

# Image registry prefix for SWE-bench Docker images (Tencent TCR).
_IMAGE_PREFIX = "swebenchdocker.tencentcloudcr.com/swebench"


def _instance_id_to_image(instance_id: str) -> str:
    """Convert a SWE-bench instance_id to the corresponding Docker image name.

    Rules (from swebench batch_instances.py):
      - Replace ``__`` with ``_1776_``
      - Lowercase the entire string
    """
    escaped = instance_id.replace("__", "_1776_").lower()
    return f"{_IMAGE_PREFIX}/sweb.eval.x86_64.{escaped}:latest"


def build_dataset(subset: str = "verified", split: str = "test") -> None:
    # Map subset names to HuggingFace dataset IDs.
    subset_map = {
        "verified": "princeton-nlp/SWE-bench_Verified",
        "lite": "princeton-nlp/SWE-bench_Lite",
        "full": "princeton-nlp/SWE-bench",
    }
    if subset not in subset_map:
        raise ValueError(f"Unknown subset '{subset}'. Choose from: {list(subset_map.keys())}")

    hf_dataset_id = subset_map[subset]
    dataset_id = f"SWEBench_{subset.capitalize()}"

    # Check if already imported.
    with Session(engine) as session:
        existing = session.exec(select(DatasetSample).where(DatasetSample.dataset == dataset_id)).all()
        if len(existing) > 0:
            print(f"Dataset '{dataset_id}' already exists ({len(existing)} samples). Skip.")
            return

    print(f"Loading {hf_dataset_id} split={split} ...")
    ds = datasets.load_dataset(hf_dataset_id, split=split)

    data: list[DatasetSample] = []
    for i, row in enumerate(ds):
        instance_id = row["instance_id"]
        data.append(
            DatasetSample(
                dataset=dataset_id,
                index=i + 1,
                source="SWEBench",
                source_index=i + 1,
                question=row["problem_statement"],
                answer="",  # SWE-bench has no text answer
                meta={
                    "instance_id": instance_id,
                    "repo": row.get("repo", ""),
                    "base_commit": row.get("base_commit", ""),
                    "image_name": _instance_id_to_image(instance_id),
                    "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
                    "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
                    "patch": row.get("patch", ""),
                },
            )
        )

    print(f"Total {len(data)} samples")
    print(f"Sample: {json.dumps(data[0].model_dump(), ensure_ascii=False, default=str)}")

    with Session(engine) as session:
        session.add_all(data)
        session.commit()
        print(f"Uploaded {len(data)} samples to dataset '{dataset_id}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import SWE-bench dataset")
    parser.add_argument(
        "--subset",
        type=str,
        default="verified",
        choices=["verified", "lite", "full"],
        help="SWE-bench subset to import.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test/dev).")
    args = parser.parse_args()
    build_dataset(subset=args.subset, split=args.split)
