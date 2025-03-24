from simple_nn import SimpleNN
from process_data import process_data
from simple_nn import generate_text

model = SimpleNN()
new_data = process_data()
model.add_data(new_data)
model.load_embeddings("embeddings_new.json")
model.load_model("second_nn")

print("ask model:")
text = input()
while len(text) != model.context_len:
    print(generate_text(text, model, 200))
    text = input()
