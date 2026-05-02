# src/finetuning/dataset.py

import torch
from torch.utils.data import Dataset


class ASAGDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        ref   = str(row["reference_answer"])
        stu   = str(row["student_answer"])
        score = float(row["score_normalized"])

        encoding = self.tokenizer(
            ref, stu,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score":          torch.tensor(score, dtype=torch.float),
            "index":          idx
        }


class T5ASAGDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        ref   = str(row["reference_answer"])
        stu   = str(row["student_answer"])
        score = float(row["score_normalized"])

        input_text = f"score answer: {ref} student: {stu}"
        encoding   = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score":          torch.tensor(score, dtype=torch.float),
            "index":          idx
        }