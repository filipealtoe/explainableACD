import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.models.anomaly_detection import run_anomaly_detection_from_files


def main() -> None:
    artifacts_dir = repo_root / "mlruns" / "4b7e95a391254fdaa79a1fb34c6a4a55" / "artifacts"

    timeseries_path = artifacts_dir / "topic_timeseries_H.csv"
    keywords_path = artifacts_dir / "topic_keywords.json"

    top_topics = [74, 0, 1, 2, 3, 4, 5, 6, 7, 8]

    with open(keywords_path, encoding="utf-8") as f:
        topic_keywords = {int(k): v for k, v in json.load(f).items()}

    run_id = run_anomaly_detection_from_files(
        timeseries_path=str(timeseries_path), top_topics=top_topics, topic_keywords=topic_keywords, z_threshold=3.0
    )

    print(f"\n{'=' * 60}")
    print("Anomaly detection complete!")
    print(f"MLflow Run ID: {run_id}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
