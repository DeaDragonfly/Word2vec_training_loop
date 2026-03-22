#!pip install python-docx
import re
import numpy as np
import random
from docx import Document
import json
from collections import Counter

#load the training corpus from a Word document
doc = Document('seals.docx')

#merge all paragraphs into one text string
text = " ".join(paragraph.text for paragraph in doc.paragraphs)
#basic stop-word list to reduce noise from very frequent function words
stop_words = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those",
    "which", "while", "when",
    "their", "they", "them",
    "it", "its", "as", "at", "by", "from", "than", "also", "both", 
    "many", "other", "others", "where", "not", "common", "important"
}

#split the corpus into sentences and normalize to lowercase
sentences = re.split(r"[.!?]+", text.lower())
sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
tokenized_sentences = [re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", sentence) for sentence in sentences] #tokenize each sentence while preserving hyphenated words
#remove stop words and one-letter tokens
tokenized_sentences = [[tok for tok in sent if tok not in stop_words and len(tok) > 1] for sent in tokenized_sentences]
tokenized_sentences = [s for s in tokenized_sentences if s]

all_tokens = [tok for sent in tokenized_sentences for tok in sent] #flatten tokenized sentences to count token frequencies

#count word frequencies and keep only words that appear at least min_count (=2) times
word_counts = Counter(all_tokens)
min_count = 2
allowed_words = {word for word, count in word_counts.items() if count >= min_count}

#remove rare words and keep only sentences that still contain at least two tokens
tokenized_sentences = [
    [tok for tok in sent if tok in allowed_words]
    for sent in tokenized_sentences
]
tokenized_sentences = [sent for sent in tokenized_sentences if len(sent) >= 2]

all_tokens = [tok for sent in tokenized_sentences for tok in sent] #rebuild the flattened token list after filtering

#build vocabulary mappings
unique_words = sorted(set(all_tokens))
word_to_id = {word: i for i, word in enumerate(unique_words)}
id_to_word = {i: word for word, i in word_to_id.items()}

sentence_ids = [[word_to_id[tok] for tok in sent] for sent in tokenized_sentences] #convert tokenized sentences into lists of integer word ids# Convert tokenized sentences into lists of integer word ids

#training hyperparameters
learning_rate = 0.025
num_negatives = 5
vocab_size = len(unique_words)
embedding_size = 50
epochs = int(input("Enter number of epochs: "))
window_size = 2

#input and output embedding matrices
W_in = np.random.randn(vocab_size, embedding_size) * 0.01
W_out = np.random.randn(vocab_size, embedding_size) * 0.01

def sigmoid(x):
   #Clip values for numerical stability
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sample_negatives(forbidden):
  #sample negative word ids that are different from the current center/context ids
    neg_ids = []
    while len(neg_ids) < num_negatives:
        neg_id = random.randrange(vocab_size)
        if neg_id not in forbidden and neg_id not in neg_ids:
          neg_ids.append(neg_id)
    return np.array(neg_ids, dtype = np.int64)

def generate_pairs(sentence_ids, window_size):
  #Generate skip-gram training pairs within each sentence
  pairs = []
  for sent in sentence_ids:
    for i, center in enumerate(sent):
      left = max(0, i - window_size)
      right = min(len(sent), i + window_size + 1)
      for j in range(left, right):
        if j != i:
          pairs.append((center, sent[j]))
  if len(pairs) == 0:
    raise ValueError("No training pairs were generated.")
  return pairs

pairs = generate_pairs(sentence_ids, window_size)

#print basic corpus statistics before training
print("Vocabulary size:", vocab_size)
print("Number of sentences:", len(sentence_ids))
print("Number of training pairs:", len(pairs))

for epoch in range (epochs):

  random.shuffle(pairs)
  total_loss = 0.0
  for w, c in pairs:
        neg_ids = sample_negatives({w,c})

        #current center word vector, positive context vector, and negative sample vectors
        center_vector = W_in[w].copy()
        positive_vector = W_out[c].copy()
        negative_vectors = W_out[neg_ids].copy()

        #forward pass: compute scores for positive and negative pairs
        s_positive = np.dot(center_vector, positive_vector)
        s_negative = np.dot(negative_vectors, center_vector)

        #convert scores to probabilities
        p_positive = sigmoid(s_positive)
        p_negative = sigmoid(s_negative)
        
        #negative sampling loss (maximize positive pair probability and minimize negative pair probabilities)
        loss = -np.log(p_positive+1e-10) - np.sum(np.log(1 - p_negative + 1e-10))
        #gradients for the center vector, positive context vector, and negative vectors
        gradient_vector = (p_positive - 1) * positive_vector + np.sum(p_negative [:, None] * negative_vectors, axis=0)
        gradient_positive_vector = (p_positive-1) * center_vector
        gradient_negative_vectors = p_negative[:, None] * center_vector

        #parameter updates
        W_in[w] -= learning_rate * gradient_vector
        W_out[c] -= learning_rate * gradient_positive_vector
        W_out[neg_ids] -= learning_rate * gradient_negative_vectors
        total_loss += loss

  print(f'Epoch {epoch +1}, loss: {total_loss / len(pairs):.6f}') #average loss per training pair for the current epoch

def most_similar(word, top_n=10):
  #average loss per training pair for the current epoch
  if word not in word_to_id:
      print(f"Word '{word}' not found in the vocabulary.")
      return []
  embeddings = (W_in + W_out) / 2

  word_id = word_to_id[word]
  word_vector = embeddings[word_id]
  similarities = []
  for i in range(len(embeddings)):
    if i == word_id:
      continue
    u = embeddings[i]
    similarity = np.dot(word_vector, u) / (np.linalg.norm(word_vector) * np.linalg.norm(u) + 1e-10)
    similarities.append((similarity, id_to_word[i]))
  similarities.sort(key = lambda x: x[0], reverse=True)
  results = []
  for sim, w in similarities[:top_n]:
    results.append((round(float(sim), 3), w))
  return results

#show nearest neighbors for a few random frequent words
candidates = [w for w in word_to_id.keys() if word_counts[w] >= min_count]
num_examples = 5
random_words = random.sample(candidates, min(num_examples, len(candidates)))

print("\nRandom frequent words and their nearest neighbors:")
for test_word in random_words:
    print(f"{test_word}: {most_similar(test_word)}")

query_word = input("Enter a word to find the 10 most similar words to it: ").lower() #interactive query for nearest neighbors
print(most_similar(query_word))

#save learned parameters
np.save("W_in.npy", W_in)
np.save("W_out.npy", W_out)

#np.savetxt("W_in.txt", W_in, fmt="%.6f")
#np.savetxt("W_out.txt", W_out, fmt="%.6f")
#print("Dimension of matrix 'Weights of words' after training: ",W_out.shape)
#print("Matrix of Weights of words after the training: ", W_out[:5])

#save vocabulary mappings
with open("word_to_id.json", "w") as f:
    json.dump(word_to_id, f)

with open("vocab.json", "w") as f:
    json.dump(unique_words, f, indent =2)
