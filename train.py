import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

"""
NEXUS AI -- Model Training Script
==================================
This script reads intents.json, processes the text using NLP,
and trains a Neural Network (MLP) using scikit-learn.

How it works:
1. Load all training patterns from intents.json
2. Tokenize and lemmatize each word (break it to its root form)
3. Convert each sentence into a "bag of words" (a list of 0s and 1s)
4. Train a Multi-Layer Perceptron (MLP) neural network classifier
5. Save the trained model and vocabulary to disk
"""

import json
import pickle
import random
import numpy as np

# ── NLTK Setup ──────────────────────────────────────────────────
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

# ── Scikit-learn Neural Network ─────────────────────────────────
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

lemmatizer = WordNetLemmatizer()

# ── Load Training Data ───────────────────────────────────────────
print("[1/5] Loading intents.json...")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []        # all unique words from patterns
classes = []      # all intent tags
documents = []    # (list_of_words, tag) pairs

ignore_chars = ['?', '!', '.', ',', "'", '"', '-', '(', ')']

# ── Tokenize & Build Vocabulary ──────────────────────────────────
print("[2/5] Processing text and building vocabulary...")
for intent in intents['intents']:
    tag = intent['tag']
    if tag == 'unknown':
        continue  # skip unknown — it has no patterns

    if tag not in classes:
        classes.append(tag)

    for pattern in intent['patterns']:
        # Tokenize the pattern into individual words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, tag))

# Lemmatize and deduplicate words (ignore punctuation)
words = sorted(set([
    lemmatizer.lemmatize(w.lower())
    for w in words
    if w not in ignore_chars
]))
classes = sorted(set(classes))

print(f"   -> {len(words)} unique words in vocabulary")
print(f"   -> {len(classes)} intent classes: {classes}")
print(f"   -> {len(documents)} training examples")

# ── Create Bag-of-Words Training Data ───────────────────────────
print("\n[3/5] Creating bag-of-words feature vectors...")

def sentence_to_bow(sentence_words):
    """Convert a list of words into a bag-of-words vector."""
    sentence_lemmas = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return [1 if w in sentence_lemmas else 0 for w in words]

X_train = []  # feature vectors
y_train = []  # labels

for (sentence_words, tag) in documents:
    bow = sentence_to_bow(sentence_words)
    X_train.append(bow)
    y_train.append(tag)

X_train = np.array(X_train)
y_train = np.array(y_train)

# ── Train Neural Network ─────────────────────────────────────────
print("\n[4/5] Training the Neural Network (MLP Classifier)...")
print("   Architecture: Input -> 128 neurons -> 64 neurons -> Output")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers for better learning
    activation='relu',                   # ReLU activation
    solver='adam',                       # Adam optimizer
    max_iter=1000,                       # train for longer
    random_state=42,
    learning_rate_init=0.001,
    verbose=False,
)

model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
print(f"\nTraining complete!")
print(f"   Training Accuracy: {train_accuracy * 100:.1f}%")
print(f"   Iterations run: {model.n_iter_}")

# ── Save Model & Vocabulary ──────────────────────────────────────
print("\n[5/5] Saving model and vocabulary...")
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

with open('intents_data.pkl', 'wb') as f:
    pickle.dump(intents, f)

print("   OK  chatbot_model.pkl  -- Trained neural network")
print("   OK  words.pkl          -- Vocabulary")
print("   OK  classes.pkl        -- Intent classes")
print("   OK  intents_data.pkl   -- Responses data")
print("\nSYU AI is trained and ready! Run app.py to start the chatbot.")
