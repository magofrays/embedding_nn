import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from process_data import save_data

print("start")
save_data()