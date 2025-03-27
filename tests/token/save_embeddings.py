import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from word2vec import word2vec
from process_data import load_tokenized_data

learning_data = load_tokenized_data()

model100 = word2vec(100, 6, 2)
model100.add_data(learning_data)
model100.train()
model100.save_embeddings("../../src_data/token/emb_token_100.json")

model500 = word2vec(500, 6, 2)
model500.add_data(learning_data)
model500.train()
model500.save_embeddings("../../src_data/token/emb_token_500.json")


model1000 = word2vec(1000, 6, 2)
model1000.add_data(learning_data)
model1000.train()
model1000.save_embeddings("../../src_data/token/emb_token_1000.json")

