from simple_nn import SimpleNN
from process_data import process_data
from process_data import clean_word
from simple_nn import generate_text

import numpy as np

def text_re(text, l):
    global learn_data
    new_data = np.array(list(clean_word(text).split()))
    word_alg = word2vec(100, l, 2)
    word_alg.add_data(learn_data)
    word_alg.add_data(new_data, True, True)
    word_alg.learn(True)
    return word_alg

def predict(model, text, l, d):
    word_alg = text_re(text, l)
    word_alg.save_embeddings(f"word2vec_new{l}.json")
    word_alg.save_word_count(f"word2vec_new_count{l}.json")

    model.load_embeddings(f"word2vec_new{l}.json")
    model.load_word_count(f"word2vec_new_count{l}.json")
    print(f"l = {l}; d = {d}")
    print(generate_text(text, model, 2))

model1 = SimpleNN()
learn_data = process_data()
model1.add_data(learn_data)
model1.load_embeddings("embeddings_new.json")
model1.load_word_count("word_count.json")
model1.load_model("first_nn")

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

predict(model1, text, 6, 100)
# predict(model1, text, 6, 500)
# predict(model1, text, 6, 1000)
