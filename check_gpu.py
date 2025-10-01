# check_gpu.py
import torch

print(f"Wersja PyTorch: {torch.__version__}")
print(f"Czy CUDA jest dostępne? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}")
    print(f"Nazwa GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("\nNiestety, PyTorch nie widzi Twojego GPU.")
    print("Możliwe przyczyny:")
    print("1. Zainstalowana jest wersja PyTorch tylko dla CPU.")
    print("2. Sterowniki NVIDIA nie są zainstalowane lub są nieaktualne.")
    print("3. Wersja CUDA Toolkit zainstalowana na komputerze jest niekompatybilna z Twoją wersją PyTorch.")