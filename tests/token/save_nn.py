import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from simple_nn import SimpleNN
from process_data import load_tokenized_data

model = SimpleNN()
new_data = load_tokenized_data()
model.add_data(new_data)
model.load_embeddings("../../src_data/token/emb_token_100.json", True)
model.compile()
model.train(5)
model.save_model("../../src_data/token/token_100_nn.keras")
