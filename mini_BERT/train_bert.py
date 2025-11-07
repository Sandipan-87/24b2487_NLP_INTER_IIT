# train/train_bert.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.bert_encoder import MiniBERT
from models.heads import MLMHead, NSPHead


def train(model, mlm_head, nsp_head, dataloader, tokenizer, config, device):
    model.to(device); mlm_head.to(device); nsp_head.to(device)
    params = list(model.parameters()) + list(mlm_head.parameters()) + list(nsp_head.parameters())
    optimizer = optim.AdamW(params, lr=config["lr"])
    loss_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    loss_nsp = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train(); mlm_head.train(); nsp_head.train()
        running = 0.0; total_nsp = 0; correct_nsp = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)
            nsp_labels = batch["nsp_labels"].to(device)

            optimizer.zero_grad()
            enc = model(input_ids, token_type_ids, attention_mask)  # (B,L,H)
            mlm_logits = mlm_head(enc)                              # (B,L,V)
            cls_hidden = enc[:, 0, :]                               # (B,H)
            nsp_logits = nsp_head(cls_hidden)                       # (B,2)

            l_mlm = loss_mlm(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1))
            l_nsp = loss_nsp(nsp_logits, nsp_labels)
            loss = l_mlm + l_nsp
            loss.backward()
            optimizer.step()

            running += loss.item()
            preds = torch.argmax(nsp_logits, dim=-1)
            total_nsp += nsp_labels.size(0)
            correct_nsp += (preds == nsp_labels).sum().item()
            pbar.set_postfix({"loss": f"{running/(pbar.n+1):.4f}", "nsp_acc": f"{correct_nsp/total_nsp:.2f}"})

        print(f"Epoch {epoch+1}: loss={running/len(dataloader):.4f}, NSP acc={correct_nsp/total_nsp:.3f}")
        demo_masked_predictions(model, mlm_head, dataloader, tokenizer, device)


def demo_masked_predictions(model, mlm_head, dataloader, tokenizer, device, n=2):
    model.eval(); mlm_head.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = mlm_head(model(input_ids, token_type_ids, attention_mask))  # (B,L,V)
        pred_ids = logits.argmax(dim=-1).cpu()

        for i in range(min(n, input_ids.size(0))):
            masked = tokenizer.decode(batch["input_ids"][i], skip_special_tokens=False)
            predicted = tokenizer.decode(pred_ids[i], skip_special_tokens=False)
            original = tokenizer.decode(batch["orig_input_ids"][i], skip_special_tokens=False)
            print("----")
            print("Masked:    ", masked)
            print("Predicted: ", predicted)
            print("Original:  ", original)
