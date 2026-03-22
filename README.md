# word2vec_training_loop

Implementation of the **Word2Vec training loop** from scratch using pure **NumPy** (no PyTorch / TensorFlow).

The goal of this project is to understand how word embeddings are learned: from raw text to vector representations via gradients and optimization.

---

## Overview

This project implements **skip-gram with negative sampling**.

The pipeline includes:
- text preprocessing
- vocabulary construction with frequency filtering
- training pair generation
- negative sampling
- forward pass
- loss computation
- gradient calculation
- parameter updates (SGD)
- similarity search using cosine similarity

---

## How it works

For each word in a sentence:
- the word is treated as a **center word**
- nearby words (within a window) are treated as **context words**
- random words are sampled as **negative examples**

The model learns to:
- increase similarity between real context pairs
- decrease similarity between random pairs

---

## Dataset

The model is trained on a custom `.docx` corpus about seals and pinnipeds.

Preprocessing steps:
- lowercase conversion
- sentence splitting
- regex-based tokenization (supports hyphenated words)
- stop-word removal
- filtering rare words (`min_count = 2`)

---

## Training configuration

Default parameters:

- embedding size: `50`
- window size: `2`
- negative samples: `5`
- learning rate: `0.025`
- Number of epochs is provided at runtime.
  
---

## Example training output:


Epoch 1, loss: 3.69. 
Epoch 2, loss: 2.30.
Epoch 3, loss: 1.11.


Loss decreases as expected.

---

## Results

After training, you can query similar words:

seals → sea, mammals, walruses, lions
behavior → feeding, mating, territory


Results reflect contextual similarity within the dataset.

---

## Output files

- `W_in.npy` — input embeddings  
- `W_out.npy` — output embeddings  
- `word_to_id.json` — word → index mapping  
- `vocab.json` — list of vocabulary words  

---

## How to run

Install dependencies:

```bash
pip install numpy python-docx
```

Run the script:

```bash
python word2vec_training_loop.py
```
Enter the number of epochs when prompted.

---

## Limitations

This is a simplified educational implementation:

- small, domain-specific dataset
- uniform negative sampling (instead of frequency-based)
- no subsampling of frequent words
- no batching or optimization tricks
- evaluation is qualitative only

---

## Purpose

This project demonstrates understanding of:

- skip-gram training logic
- negative sampling
- gradient-based optimization
- how embeddings emerge from co-occurrence

The focus is on implementing the training loop manually rather than using high-level libraries.
