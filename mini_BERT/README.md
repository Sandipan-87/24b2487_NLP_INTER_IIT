# MiniBERT – Simplified BERT from Scratch (PyTorch)

This project is a **lightweight, encoder-only Transformer model** implemented from scratch in **PyTorch**, inspired by the original paper:

 **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
 *Jacob Devlin et al., 2018* ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805))

It trains on the **WikiText-2** dataset using the same two pretraining objectives as BERT:
- **Masked Language Modeling (MLM)** — predict randomly masked words.
- **Next Sentence Prediction (NSP)** — classify whether sentence B follows sentence A.

---

## Features

1. Encoder-only Transformer (no decoder)  
2. Token + Position + Segment embeddings  
3. Multi-Head Self-Attention & Feed-Forward layers  
4. Special tokens `[CLS]`, `[SEP]`, `[MASK]`  
5. Joint MLM + NSP objective  
6. Fully from scratch — no prebuilt `BERTModel` used  
7. Works on CPU or GPU  
8. Compatible with Hugging Face WikiText-2 dataset  

---



