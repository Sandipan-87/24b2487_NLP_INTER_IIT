# main.py
import argparse
import random
import yaml
import torch

from data.prepare_dataset import get_dataloader
from models.bert_encoder import MiniBERT
from models.heads import MLMHead, NSPHead
from train.train_bert import train


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg["device"] if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data/tokenizer...")
    dl, tokenizer = get_dataloader(
        tokenizer_name=cfg["vocab_name"],
        split="train",
        batch_size=cfg["batch_size"],
        max_len=cfg["max_seq_len"],
        mask_prob=cfg["mask_prob"],
        shuffle=True,
    )
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    print("Building model...")
    model = MiniBERT(
        vocab_size=vocab_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        max_position_embeddings=cfg["max_seq_len"],
        type_vocab_size=2,
        dropout=cfg["dropout"],
    )
    mlm_head = MLMHead(cfg["hidden_size"], vocab_size)
    nsp_head = NSPHead(cfg["hidden_size"])

    print("Training...")
    train(
        model=model,
        mlm_head=mlm_head,
        nsp_head=nsp_head,
        dataloader=dl,
        tokenizer=tokenizer,
        config={"lr": cfg["lr"], "epochs": cfg["epochs"]},
        device=device,
    )
