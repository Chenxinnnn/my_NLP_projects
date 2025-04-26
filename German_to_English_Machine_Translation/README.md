# 🧠 HW2: Machine Translation with Transformers

This project implements a German-to-English machine translation model using the Transformer architecture from scratch.

## 📚 Overview

This week, we built a German-English translation model from scratch using the Transformer architecture as part of a graduate-level NLP course at NYU
The project includes theoretical derivations, implementation, training, and analysis.

---

## 📁 Directory Structure

```text
code/
├── layers.py              # Transformer modules (MultiHeadAttention, PositionalEncoding, etc.)
├── main.py                # Entry point for training and evaluation
├── transformer.py         # Transformer model definition (encoder/decoder stack)
├── utils.py               # Beam search, data batching, helper functions
├── test.py                # Unit tests
├── vocab.pt               # Saved vocabulary object
├── requirements.txt       # Required packages
├── out_greedy.txt         # Translation results using greedy decoding
├── out_beam.txt           # Translation results using beam search
├── README.md              # This file
├── data/                  # Preprocessed data (Multi30k dataset)
└── __pycache__/           # Python cache files
```

---

## 📦 Data

- Dataset: [Multi30k (German-English)](https://github.com/multi30k/dataset)
- Preprocessed files are stored in `./data/`
- Vocabulary object is saved as `vocab.pt`

---

## 🔮 Maybe next week

- Add subword tokenization (e.g., SentencePiece or Byte Pair Encoding)
- Visualize attention weights for better model interpretability
- Add coverage penalty to improve beam search diversity
- Explore BLEURT or COMET as alternative evaluation metrics

---

## ✏️ Author

Chenxin Gu @ NYU Data Science  
For academic use only.
