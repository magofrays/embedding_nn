from word2vec import word2vec
from process_data import process_data

learning_data = process_data()
model = word2vec(100, 6, 2)
model.add_data(learning_data)
model.learn()
model.save_embeddings("../src_data/embeddings_new.json")