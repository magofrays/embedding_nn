import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "src")))

from process_data import load_tokenized_data, decode_text, load_data

data = load_tokenized_data()
o_data = load_data()
print(data[:100])
print(decode_text(data[:500]))
print(" ".join(o_data[:100]))