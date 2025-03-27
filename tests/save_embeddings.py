import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from word2vec import word2vec
from process_data import load_data

learning_data = load_data()
# with open ('../src_data/data.txt', 'w', encoding='utf-8') as f:
#     f.write(' '.join(learning_data))
model = word2vec(100, 6, 2)
model.add_data(learning_data)
model.save_word_count("../src_data/word_count.json")
# model.train()
# model.save_embeddings("../src_data/embeddings_new.json")