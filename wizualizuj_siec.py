
# wizualizuj_siec.py
# Skrypt do analizy i wizualizacji wewnętrznej struktury sieci modelu "Mini-Smoczątko".

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

# Importujemy potrzebne elementy z pliku treningowego
try:
    from mini_smoczatko import BDH_GPU, D, H, N
except ImportError:
    print("Błąd: upewnij się, że plik 'mini_smoczatko.py' jest w tym samym folderze.")
    sys.exit(1)

MODEL_PATH = 'mini_smoczatko.pth'
OUTPUT_IMAGE_FILE = 'analiza_wynik.png'

def visualize():
    # --- Krok 1: wczytaj wytrenowany model ---
    try:
        model = BDH_GPU()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"Pomyślnie wczytano model z pliku '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Błąd: nie znaleziono pliku modelu '{MODEL_PATH}'.")
        print("Najpierw uruchom 'mini_smoczatko.py', aby wytrenować i zapisać model.")
        return
        
    # --- Krok 2: wyciągnij macierz 'encoder' ---
    encoder = model.encoder.detach().cpu().numpy()
    print("Wyciągnięto macierz 'encoder' z modelu.")

    # --- Krok 3: oblicz graf podobieństwa neuronów ---
    G = encoder @ encoder.T
    print(f"Obliczono macierz podobieństwa neuronów G o wymiarach: {G.shape}")

    # --- Krok 4: przygotuj dane do wizualizacji ---
    weights = G[~np.eye(G.shape[0], dtype=bool)].flatten()
    print(f"Analizowanie {len(weights)} unikalnych wag połączeń...")
    
    # --- Krok 5: stwórz i zapisz wykres ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.hist(weights, bins=200, log=True, color='purple', alpha=0.75, label='Częstotliwość wag')
    ax.set_title('Dystrybucja podobieństw w sieci neuronowej (skala logarytmiczna)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Wartość podobieństwa (iloczyn skalarny)', fontsize=12)
    ax.set_ylabel('Liczba wystąpień (w skali log)', fontsize=12)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero (brak korelacji)')
    
    mean_weight = np.mean(weights)
    ax.axvline(mean_weight, color='green', linestyle=':', linewidth=2, label=f'Średnia ({mean_weight:.4f})')
    
    ax.legend()
    fig.tight_layout()
    
    try:
        plt.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight')
        print(f"\n>>> Pomyślnie zapisano wykres do pliku: '{OUTPUT_IMAGE_FILE}' <<<")
    except Exception as e:
        print(f"Błąd podczas zapisywania wykresu: {e}")

    # Opcjonalnie: pokaż wykres na ekranie
    plt.show()

if __name__ == '__main__':
    visualize()