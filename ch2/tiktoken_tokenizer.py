# OpenAI's byte-pair encoding tokenizing library
# https://github.com/openai/tiktoken
#
# BPE breaks words down into subsequences (not necessarily syllables)
# so that it can process words that it's never seen before
from dataclasses import dataclass

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")


def sanity_checks():
    text = """The quick brown fox jumps over the lazy dog"""
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text


# Ingest sample text
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

@dataclass
class GPTDatasetV1(Dataset):
    input_ids: list[torch.Tensor]
    target_ids: list[torch.Tensor]

    @classmethod
    def from_text(cls, text, tokenizer, window_length, stride):
        input_ids = []
        target_ids = []

        token_ids = tokenizer.encode(text)

        # Use a sliding window to chunk the book into overlapping sequences
        for i in range(0, len(token_ids) - window_length, stride):
            input_chunk = token_ids[i : i + window_length]
            target_chunk = token_ids[i + 1 : i + window_length + 1]

            input_ids.append(torch.tensor(input_chunk))
            target_ids.append(torch.tensor(target_chunk))

        return GPTDatasetV1(input_ids, target_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    window_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1.from_text(txt, tokenizer, window_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
