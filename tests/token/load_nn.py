import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from simple_nn import SimpleNN
from process_data import load_tokenized_data, decode_text, encode_text
from simple_nn import generate_text
from word2vec import word2vec

import numpy as np

def text_re(text, l):
    global learn_data
    word_alg = word2vec(100, l, 2)
    word_alg.add_data(learn_data)
    word_alg.add_data(text, True, True)
    word_alg.train(True)
    return word_alg

def predict(model, text, l, d):
    word_alg = text_re(text, l)
    word_alg.save_embeddings(f"../../src_data/token/word2vec_new{d}.json")
    word_alg.save_word_count(f"../../src_data/token/word2vec_new_count{d}.json")
    model.load_embeddings(f"../../src_data/token/word2vec_new{d}.json")
    model.load_word_count(f"../../src_data/token/word2vec_new_count{d}.json")
    print(f"l = {l}; d = {d}")
    gen_data = generate_text(text, model, 30)
    print(gen_data)
    gen_text = decode_text(gen_data)
    print(gen_text)
    

model100 = SimpleNN()
learn_data = load_tokenized_data()
model100.add_data(learn_data)
model100.load_embeddings("../../src_data/token/emb_token_100.json")
model100.load_word_count("../../src_data/token/word_count_100.json")
model100.load_model("../../src_data/token/token_100_nn.keras")

# model2 = SimpleNN()
# model2.add_data(learn_data)
# model2.load_embeddings("embeddings_new.json")
# model2.load_word_count("word_count.json")
# model2.load_model("first_nn")

# model3 = SimpleNN()
# model3.add_data(learn_data)
# model3.load_embeddings("embeddings_new.json")
# model3.load_word_count("word_count.json")
# model3.load_model("first_nn")

print("ask model:")

text = input()
text = encode_text(text)

predict(model100, text, 6, 100)
# predict(model1, text, 6, 500)
# predict(model1, text, 6, 1000)
