import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simple_nn import SimpleNN
from process_data import load_data

model = SimpleNN()
new_data = load_data()
model.add_data(new_data)
model.load_embeddings("../src_data/embeddings_new.json")
model.compile()
model.train(50, False)
model.save_model("../src_data/second_nn.keras")
