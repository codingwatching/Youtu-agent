"""Export SWE-bench patches from evaluation DB to JSONL for offline evaluation.

Reads rollout results (where ``extracted_final_answer`` stores the git diff
patch) and writes a ``.jsonl`` file compatible with
``swebench.harness.run_evaluation --predictions_path``.

Usage:
    python scripts/data/export_swe_bench_patches.py --exp_id swe_test_01

    # Then run offline evaluation:
    python -m swebench.harness.run_evaluation \\
        --predictions_path patches.jsonl \\
        --swe_bench_tasks princeton-nlp/SWE-bench_Verified \\
        --run_id swe_test_01
"""

import argparse
import json
from pathlib import Path

from sqlmodel import select

from utu.db.eval_datapoint import EvaluationSample
from utu.utils import SQLModelUtils, get_logger

logger = get_logger(__name__, "INFO")


def export_patches(exp_id: str, output: str, model_name: str) -> None:
    with SQLModelUtils.create_session() as session:
        samples = session.exec(
            select(EvaluationSample)
            .where(
                EvaluationSample.exp_id == exp_id,
                EvaluationSample.stage.in_(["rollout", "judged"]),
            )
            .order_by(EvaluationSample.dataset_index)
        ).all()

    if not samples:
        logger.error(f"No rollout samples found for exp_id='{exp_id}'.")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    empty_patch = 0
    with open(output_path, "w") as f:
        for sample in samples:
            meta = sample.meta or {}
            instance_id = meta.get("instance_id", "")
            if not instance_id:
                logger.warning(f"Sample index={sample.dataset_index} has no instance_id in meta, skipping.")
                continue

            patch = sample.extracted_final_answer or ""
            if not patch.strip():
                empty_patch += 1

            entry = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": patch,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

    logger.info(
        f"Exported {total} predictions to {output_path} "
        f"({total - empty_patch} with patch, {empty_patch} empty)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SWE-bench patches for offline evaluation")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID to export.")
    parser.add_argument(
        "--output", type=str, default="patches.jsonl", help="Output JSONL file path (default: patches.jsonl)."
    )
    parser.add_argument(
        "--model_name", type=str, default="utu-agent", help="Model name tag in predictions (default: utu-agent)."
    )
    args = parser.parse_args()
    export_patches(exp_id=args.exp_id, output=args.output, model_name=args.model_name)
