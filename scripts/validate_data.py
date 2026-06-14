# scripts/validate_data.py
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_processed_data(data_dir: str = "data/processed/"):
    """Validate processed data files."""
    data_dir = Path(data_dir)

    for split in ["train", "val", "test"]:
        filepath = data_dir / f"{split}.json"
        if not filepath.exists():
            print(f"❌ {split}.json not found")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n=== {split.upper()} ===")
        print(f"Examples: {len(data)}")

        if data:
            sample = data[0]
            print(f"Sample question: {sample.get('question', 'N/A')[:100]}...")
            print(f"Number of choices: {len(sample.get('choices', []))}")
            print(f"Number of sessions: {sample.get('num_sessions', 0)}")
            print(f"Question type: {sample.get('question_type', 'N/A')}")

            # Check for missing data
            missing = []
            for key in ["question", "choices", "answer_index", "session_texts"]:
                if not sample.get(key):
                    missing.append(key)
            if missing:
                print(f"⚠️ Missing fields: {missing}")
            else:
                print("✅ All required fields present")

    # Check stats
    stats_file = data_dir / "stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        print(f"\n=== DATASET STATS ===")
        print(f"Total QA pairs: {stats['total_qa_pairs']}")
        print(f"Question types: {stats['question_types']}")


if __name__ == "__main__":
    validate_processed_data()