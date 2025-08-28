import sys
import platform
import torch
import pandas as pd
import sklearn

# Python ve platform bilgileri
print("Python Version:", sys.version)
print("Platform:", platform.platform())

print("\n--- Kütüphane Versiyonları ---")
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)

# GPU (CUDA) kontrolü
print("\n--- GPU Kontrolü ---")
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use GPU.")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch will use CPU.")