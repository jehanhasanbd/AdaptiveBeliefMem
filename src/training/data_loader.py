# src/training/data_loader.py (fixed)
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional
from pathlib import Path


class LoCoMoDataset(Dataset):
    """Dataset for LoCoMo conversations."""

    def __init__(self, data_path: str, max_sessions: Optional[int] = None):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)

        self.max_sessions = max_sessions
        print(f"Loaded {len(self.examples)} examples from {data_path}")

        # Filter out examples with missing data
        self.valid_examples = []
        for ex in self.examples:
            if (ex.get("session_texts") and len(ex["session_texts"]) > 0 and
                    ex.get("question") and
                    ex.get("choices") and
                    ex.get("answer_index", -1) >= 0):
                self.valid_examples.append(ex)

        print(f"Valid examples: {len(self.valid_examples)}")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        example = self.valid_examples[idx]

        # Limit sessions if specified
        sessions = example["session_texts"]
        if self.max_sessions and len(sessions) > self.max_sessions:
            sessions = sessions[-self.max_sessions:]

        return {
            "conversation_id": example.get("conversation_id", ""),
            "session_texts": sessions,
            "num_sessions": len(sessions),
            "question": example["question"],
            "choices": example.get("choices", []),
            "answer": example.get("answer", ""),
            "answer_index": example.get("answer_index", -1),
            "question_type": example.get("question_type", "unknown")
        }


class LoCoMoDataLoader:
    """Data loader for LoCoMo dataset with batching."""

    def __init__(self, data_dir: str, batch_size: int = 1, max_sessions: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.max_sessions = max_sessions

    def get_dataloader(self, split: str, shuffle: bool = False) -> DataLoader:
        """Get dataloader for train/val/test split."""
        if split == "train":
            path = self.data_dir / "train.json"
        elif split == "val":
            path = self.data_dir / "val.json"
        elif split == "test":
            path = self.data_dir / "test.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        if not path.exists():
            print(f"Warning: {path} not found, creating empty dataset")
            # Create empty file if doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump([], f)

        dataset = LoCoMoDataset(str(path), self.max_sessions)

        if len(dataset) == 0:
            print(f"Warning: {split} dataset is empty!")

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate batch of examples."""
        return {
            "conversation_ids": [b["conversation_id"] for b in batch],
            "session_texts": [b["session_texts"] for b in batch],
            "num_sessions": [b["num_sessions"] for b in batch],
            "questions": [b["question"] for b in batch],
            "choices": [b["choices"] for b in batch],
            "answers": [b["answer"] for b in batch],
            "answer_indices": [b["answer_index"] for b in batch],
            "question_types": [b["question_type"] for b in batch]
        }