from simple_nn import SimpleNN
from process_data import process_data

model = SimpleNN()
new_data = process_data()
model.add_data(new_data)
model.load_embeddings("embeddings_new.json")
model.compile()
model.train(10, False)
model.save_model("first_nn")
