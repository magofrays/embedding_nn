import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from process_data import save_data, load_data

save_data()
print(load_data)

