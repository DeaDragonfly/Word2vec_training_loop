##word2vec_training_loop

This project is a from-scratch implementation of the Word2Vec training loop using pure NumPy.

The goal was not to use any ML frameworks, but to understand how word embeddings are actually learned: from raw text → to vectors → through gradients and optimization.

What’s inside

This is a skip-gram with negative sampling implementation.

The code includes:

text loading from a .docx file
preprocessing (tokenization, stop-word removal, sentence splitting)
vocabulary construction with min_count filtering
generation of (center, context) training pairs
negative sampling
full training loop:
forward pass
loss computation
gradient calculation
parameter updates (SGD)
cosine similarity search for nearest words

No PyTorch, no TensorFlow — only NumPy.

How it works (short version)

For each word in a sentence:

take it as a center word
look at nearby words (window)
treat them as positive examples
sample random words as negative examples

The model learns to:

increase similarity between real neighbors
decrease similarity between random words
Dataset

The model is trained on a custom text corpus about seals and pinnipeds.

Preprocessing steps:

lowercase
sentence splitting
regex tokenization (supports hyphenated words)
stop-word removal
removal of rare words (min_count = 2)
Training

Main parameters:

embedding size: 50
window size: 2
negative samples: 5
learning rate: 0.025

You can choose the number of epochs when running the script.

Example output:

Epoch 1, loss: 3.69
Epoch 2, loss: 2.30
Epoch 3, loss: 1.11
...

Loss decreasing = training works as expected.

Results

After training, you can query similar words:

seals → sea, mammals, walruses, lions
behavior → feeding, mating, territory

Results are not perfect (small dataset), but they show that the model learns meaningful relationships.

Files saved
W_in.npy – input embeddings
W_out.npy – output embeddings
word_to_id.json – mapping word → index
vocab.json – list of words
Limitations

This is a simplified educational implementation:

small and domain-specific dataset
uniform negative sampling (not frequency-based)
no subsampling of frequent words
no batching or performance optimizations
evaluation is qualitative only
Why this project

The point was to understand:

how Word2Vec actually works under the hood
how gradients are derived and applied
how embeddings emerge from co-occurrence

Not just to use a library, but to build the core logic manually.
