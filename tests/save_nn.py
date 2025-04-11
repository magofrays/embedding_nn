import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simple_nn import SimpleNN
from process_data import load_data

model = SimpleNN()
new_data = load_data()
model.add_data(new_data)
model.load_embeddings("../src_data/emb_100.json")
model.compile()
model.train(10)
model.save_model("../src_data/100_nn.keras")
