import sys
import os

# Добавляем путь к папке src в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from word2vec import word2vec
from process_data import process_data

learning_data = process_data()
model = word2vec(100, 6, 2)
model.add_data(learning_data)
model.save_word_count()
# model.learn()
# model.save_embeddings("../src_data/embeddings_new.json")