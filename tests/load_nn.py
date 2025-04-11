import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simple_nn import SimpleNN
from process_data import load_data
from process_data import clean_word
from simple_nn import generate_text
from word2vec import word2vec

import numpy as np

def text_re(text, l, d):
    global learn_data
    new_data = np.array(list(clean_word(text).split()))
    word_alg = word2vec(d, l, 2)
    word_alg.load_embeddings(f"../src_data/emb_{d}.json")
    word_alg.add_data(learn_data)
    word_alg.add_data(new_data, True, True)
    word_alg.train(True)
    return word_alg

def predict(model, text, l, d):
    word_alg = text_re(text, l, d)
    word_alg.save_embeddings(f"../src_data/word2vec_new{d}.json")
    word_alg.save_word_count(f"../src_data/word2vec_new_count{d}.json")
    model.load_embeddings(f"../src_data/word2vec_new{d}.json")
    model.load_word_count(f"../src_data/word2vec_new_count{d}.json")
    print(f"l = {l}; d = {d}")
    gen_data = generate_text(clean_word(text).split(), model, 2)
    print(" ".join(gen_data))

model1 = SimpleNN()
learn_data = load_data()
model1.add_data(learn_data)

model1.load_model("../src_data/100_nn.keras")

model2 = SimpleNN()
model2.add_data(learn_data)
# model2.load_embeddings("embeddings_new.json")
# model2.load_word_count("word_count.json")
model2.load_model("../src_data/500_nn.keras")

model3 = SimpleNN()
model3.add_data(learn_data)
# model3.load_embeddings("embeddings_new.json")
# model3.load_word_count("word_count.json")
model3.load_model("../src_data/1000_nn.keras")

print("ask model:")

text = input()

predict(model1, text, 6, 100)
predict(model2, text, 6, 500)
predict(model3, text, 6, 1000)
