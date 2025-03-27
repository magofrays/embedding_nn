import numpy as np
import re
import os


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
    data = np.array(open("../src_data/data.txt").read().split())
    return data
