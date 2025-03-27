import numpy as np
import re
import os
from tokenizer import BPETokenizer

def clean_word(word):
    return re.sub(r'[^\w\s@#.,!?;:"\'\-]', '', word).strip()


def process_data():
    learning_files = os.listdir("../learning_data")
    data = np.array([], dtype=str)
    for f in learning_files:
        with open("../learning_data/" + f, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()
            cleaned_words = [clean_word(word) for word in words]
            data = np.append(data, cleaned_words)
    data = data[data != '']
    print("Data proceed successfully")
    print(data)
    return data

def save_data():
    data = process_data()
    print(data)
    with open("../src_data/data.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(data))

def load_data():
    data = np.array(open("../../src_data/data.txt").read().split())
    return data

# it is used in tests/token folder
def train_tokenizer(vocab_size):
    data = open("../../src_data/data.txt").read()
    tokenizer = BPETokenizer()
    tokenizer.train(data, vocab_size)
    tokenizer.save_vocab_and_merges("../../src_data/token/vocab.json", "../../src_data/token/merges.json")

def save_tokenized_data():
    data = open("../../src_data/data.txt").read()
    tokenizer = BPETokenizer()
    tokenizer.load_vocab_and_merges("../../src_data/token/vocab.json", "../../src_data/token/merges.json")
    tokenized_data = tokenizer.encode(data)
    tokenized_data = list(map(str, tokenized_data))
    with open("../../src_data/tokenized_data.txt", "w", encoding="utf-8") as file:
        file.write(" ".join(tokenized_data))

def load_tokenized_data():
    tokenized_data = open("../../src_data/tokenized_data.txt").read()
    tokenized_data = list(map(int, tokenized_data.split()))
    return tokenized_data

def encode_text(text):
    tokenizer = BPETokenizer()
    tokenizer.load_vocab_and_merges("../../src_data/token/vocab.json", "../../src_data/token/merges.json")
    return tokenizer.encode(text)

def decode_text(tokenized_text):
    tokenizer = BPETokenizer()
    tokenizer.load_vocab_and_merges("../../src_data/token/vocab.json", "../../src_data/token/merges.json")
    return tokenizer.decode(tokenized_text)