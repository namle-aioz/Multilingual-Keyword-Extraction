import argparse
import csv
import json
import time
from pathlib import Path

from multiple_extraction import (
    DATA_CSV,
    embed_model,
    get_or_create_index,
    load_topics_from_csv,
    normalize_to_english,
    process_multilingual,
    remove_html,
)


def run_batch(csv_path: Path, output_path: Path) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    topics_data = load_topics_from_csv(DATA_CSV)
    active_index, active_meta = get_or_create_index(topics_data)

    results = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            raw_text = (row.get("Content") or "").strip()
            language = (row.get("Language") or "").strip()
            expect_result = (row.get("Expected_Result") or "").strip()

            word_count = len(raw_text.split())
            cleaned = remove_html(raw_text)
            normalized = normalize_to_english(cleaned)

            t0 = time.perf_counter()
            response = process_multilingual(
                normalized,
                active_index,
                active_meta,
                embed_model,
                word_count=word_count,
            )
            elapsed = time.perf_counter() - t0

            results.append(
                {
                    "row": i,
                    "language": language,
                    "word_count": word_count,
                    "expected_result": expect_result,
                    "response": response,
                    "latency_seconds": round(elapsed, 4),
                }
            )

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read CSV Content column, run extraction, and write responses to JSON."
    )
    parser.add_argument(
        "--input",
        default="tescase.csv",
        help="Input CSV file path (default: tescase.csv)",
    )
    parser.add_argument(
        "--output",
        default="tescase_results.json",
        help="Output JSON file path (default: tescase_results.json)",
    )

    args = parser.parse_args()
    csv_path = Path(args.input)
    output_path = Path(args.output)

    count = run_batch(csv_path, output_path)
    print(f"Wrote {count} results to {output_path}")


if __name__ == "__main__":
    main()
