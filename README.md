# BERT from Scratch (1 Epoch, Training Loss: 4.13)

These are the scripts for creating the BERT model trained from scratch using a custom tokenizer with a 64,000-token vocabulary. The model can be found here: https://huggingface.co/ishathombre/monolingual-hindi-from-scratch

- **Training:** 1 epoch
- **Masked Language Modeling (MLM) loss:** 4.13
- **Tokenizer:** Custom-trained, vocab size, on iit-madras Hindi-monolingual dataset = 64,000
- **Architecture:**
Maximum position embeddings: 512
Hidden size: 312
Number of attention heads: 12
Number of transformer layers: 4
Intermediate (feed-forward) size: 1200
Type vocabulary size: 2 (for segment embeddings)

It is uploaded for checkpointing, experimentation, and community feedback.

## Intended Use

- Research on training dynamics
- Continued pretraining
- Fine-tuning for downstream tasks (with caution)

## Limitations

- Low training coverage (1 epoch)
- Not yet evaluated on downstream tasks

  
