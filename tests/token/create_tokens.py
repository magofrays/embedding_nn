import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from process_data import save_data, train_tokenizer, save_tokenized_data

save_data()
train_tokenizer(10000)
save_tokenized_data()

