import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .arguments import DataArguments

@dataclass
class RerankCollator:
    tokenizer: PreTrainedTokenizer
    query_max_len: int = 32
    passage_max_len: int = 128
    pad_to_multiple_of: int = 8
    return_tensors: str = "pt"
    padding: bool = True

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        query = [f["query"] for f in features]
        passage = [f["passage"] for f in features]

        batch_query = self.tokenizer(
            query,
            max_length=self.query_max_len,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch_passage = self.tokenizer(
            passage,
            max_length=self.passage_max_len,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch = {
            "query_input_ids": batch_query.input_ids,
            "query_attention_mask": batch_query.attention_mask,
            "passage_input_ids": batch_passage.input_ids,
            "passage_attention_mask": batch_passage.attention_mask,
        }

        if "label" in features[0]:
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)

        return batch

class TrainDatasetForReranker(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.total = 0
        self.datasets = self.load_data()

    def load_data(self):
        data = []
        with open(self.args.train_data, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.total = len(data)
        return data

    def __len__(self):
        return self.total

    def __getitem__(self, item):
        group = self.datasets[item]
        query = group['query']
        group_size = min(self.args.train_group_size, len(group['pos']) + len(group['neg']))
        if len(group['pos']) > 0:
            pos = random.choice(group['pos'])
            neg = random.choices(group['neg'], k=group_size - 1) if group_size > 1 else []
            passages = [pos] + neg
        else:
            passages = random.choices(group['neg'], k=group_size)
        random.shuffle(passages)
        return {
            "query": self.args.query_instruction_for_retrieval + query,
            "passage": [self.args.passage_instruction_for_retrieval + p for p in passages],
            "label": [1 if p == pos else 0 for p in passages] if len(group['pos']) > 0 else [0] * group_size
        }

class EvalDatasetForReranker(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.total = 0
        self.datasets = self.load_data()

    def load_data(self):
        data = []
        with open(self.args.eval_data, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        self.total = len(data)
        if self.args.max_eval_samples is not None:
            self.total = min(self.total, self.args.max_eval_samples)
        return data[:self.total]

    def __len__(self):
        return self.total

    def __getitem__(self, item):
        group = self.datasets[item]
        query = group['query']
        passages = group['passages']
        labels = group['labels'] if 'labels' in group else [0] * len(passages)
        
        return {
            "query": self.args.query_instruction_for_retrieval + query,
            "passage": [self.args.passage_instruction_for_retrieval + p for p in passages],
            "label": labels
        }
