from simple_nn import SimpleNN
from process_data import process_data

model = SimpleNN()
new_data = process_data()
model.add_data(new_data)
model.load_embeddings("embeddings_new.json")
model.compile()
model.train(50, False)
model.save_model("second_nn")
