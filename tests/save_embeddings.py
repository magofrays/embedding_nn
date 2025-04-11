import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from word2vec import word2vec
from process_data import load_data

learning_data = load_data()
model = word2vec(100, 6, 2)
model.add_data(learning_data)
model.save_word_count("../src_data/word_count_100.json")
model.train()
model.save_embeddings("../src_data/emb_100.json")