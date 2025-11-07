# data/prepare_data.py
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast


class NSPDataset(Dataset):
    """
    Builds (A, B, is_next) pairs:
      - positives: consecutive sentences (is_next=1)
      - negatives: A with random B (is_next=0)
    """
    def __init__(self, sentences: List[str]):
        pairs = []
        for i in range(len(sentences) - 1):
            a = sentences[i].strip()
            b = sentences[i + 1].strip()
            if a and b:
                pairs.append((a, b, 1))
        # add one negative per positive
        all_sents = [s for s in sentences if s.strip()]
        for (a, _, _) in list(pairs):
            b_neg = random.choice(all_sents)
            pairs.append((a, b_neg, 0))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        return {"a": a, "b": b, "is_next": y}


def _split_to_sentences(raw_lines: List[str]) -> List[str]:
    # WikiText is line-based; many lines are already sentence-ish.
    # We keep non-empty lines; simple and robust for this assignment.
    sents = []
    for t in raw_lines:
        if t and t.strip() and not t.strip().startswith("= "):
            sents.append(t.strip())
    return sents


def load_wikitext2_sentences(split: str = "train") -> List[str]:
    ds = load_dataset("wikitext", "wikitext-2-v1", split=split)
    raw = [row["text"] for row in ds]
    return _split_to_sentences(raw)


def mask_tokens(input_ids: torch.Tensor, tokenizer: BertTokenizerFast, mask_prob: float = 0.15):
    """
    Standard BERT masking:
      - choose 15% of tokens (not specials)
      - of those:
          80% -> [MASK]
          10% -> random token
          10% -> keep same
    Returns masked_input_ids and mlm_labels (non-masked => -100).
    """
    labels = input_ids.clone()
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix = torch.full(labels.shape, mask_prob)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    # 80% -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% -> random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% remain unchanged
    return input_ids, labels


def _collate_fn(
    batch: List[Dict],
    tokenizer: BertTokenizerFast,
    max_len: int,
    mask_prob: float,
):
    text_a = [x["a"] for x in batch]
    text_b = [x["b"] for x in batch]
    is_next = torch.tensor([x["is_next"] for x in batch], dtype=torch.long)

    enc = tokenizer(
        text_a,
        text_b,
        padding="longest",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"]
    token_type_ids = enc["token_type_ids"]
    attention_mask = enc["attention_mask"]

    masked_input_ids, mlm_labels = mask_tokens(input_ids.clone(), tokenizer, mask_prob)
    return {
        "input_ids": masked_input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "mlm_labels": mlm_labels,
        "nsp_labels": is_next,
        "orig_input_ids": input_ids,  # for pretty printing
    }


def get_dataloader(
    tokenizer_name: str,
    split: str,
    batch_size: int,
    max_len: int,
    mask_prob: float,
    shuffle: bool = True,
):
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    sentences = load_wikitext2_sentences(split)
    dataset = NSPDataset(sentences)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: _collate_fn(b, tokenizer, max_len, mask_prob),
    )
    return dl, tokenizer
