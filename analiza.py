# analiza.py
# Kompletny skrypt do wizualizacji struktury sieci w wytrenowanym modelu "Mini-Smoczątko".

import torch
import matplotlib.pyplot as plt
import numpy as np # Dodajemy import numpy dla lepszej obsługi danych

# Ważne: Importujemy klasę modelu ORAZ parametry z naszego pliku treningowego.
# To gwarantuje, że model zostanie wczytany z poprawną architekturą.
try:
    from mini_smoczatko import BDH_GPU, D, H, N
except ImportError:
    print("Błąd: Nie można zaimportować modelu z pliku 'mini_smoczatko.py'.")
    print("Upewnij się, że oba pliki ('analiza.py' i 'mini_smoczatko.py') są w tym samym folderze.")
    exit()

MODEL_PATH = 'mini_smoczatko.pth'

def main():
    """Główna funkcja skryptu."""
    
    # --- Krok 1: Wczytaj wytrenowany model ---
    try:
        model = BDH_GPU()
        # map_location='cpu' pozwala na analizę nawet na komputerze bez GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print(f"Pomyślnie wczytano wytrenowany model z pliku '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku modelu '{MODEL_PATH}'.")
        print("Czy na pewno zakończyłeś trening, uruchamiając 'mini_smoczatko.py'?")
        return
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas wczytywania modelu: {e}")
        return

    # --- Krok 2: Wyciągnij kluczową macierz wag ---
    # Wyciągamy macierz 'encoder', która mapuje z przestrzeni koncepcyjnej N do przestrzeni roboczej D.
    # Każdy wiersz tej macierzy to "reprezentacja wyjściowa" jednego neuronu.
    # Używamy .detach().cpu().numpy() aby przekształcić tensor w tablicę NumPy do analizy.
    encoder = model.encoder.detach().cpu().numpy()
    print("Wyciągnięto macierz 'encoder' z modelu.")

    # --- Krok 3: Oblicz graf podobieństwa neuronów ---
    # Obliczymy macierz G = encoder @ encoder.T.
    # Wynikowa macierz G (N x N) reprezentuje graf podobieństwa między wszystkimi neuronami.
    # Wartość G[i, j] jest iloczynem skalarnym wektorów dla neuronu i oraz j.
    # Wysoka wartość oznacza, że neurony te mają podobne "role" w sieci.
    # Jest to uproszczona, ale bardzo skuteczna metoda na zwizualizowanie struktury.
    
    G = encoder @ encoder.T
    print(f"Obliczono macierz podobieństwa neuronów G o wymiarach: {G.shape}")

    # --- Krok 4: Analiza i wizualizacja dystrybucji wag ---
    # Spłaszczamy macierz 2D do jednego, długiego wektora wag/podobieństw.
    # Pomijamy wartości na diagonali, ponieważ podobieństwo neuronu do samego siebie nie jest interesujące.
    weights = G[~np.eye(G.shape[0], dtype=bool)].flatten()

    print(f"Analizowanie {len(weights)} wag połączeń...")
    
    # Ustawiamy styl wykresu na bardziej czytelny
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    
    # Używamy histogramu z logarytmiczną skalą na osi Y, aby zobaczyć "ciężkie ogony"
    plt.hist(weights, bins=200, log=True, color='purple', alpha=0.75, label='Częstotliwość wag')
    
    plt.title('Dystrybucja Podobieństw w Sieci Neuronowej (Skala Logarytmiczna)', fontsize=16, fontweight='bold')
    plt.xlabel('Wartość Podobieństwa (Iloczyn Skalarny)', fontsize=12)
    plt.ylabel('Liczba Wystąpień (w skali log)', fontsize=12)
    
    # Dodajemy linię w zerze dla lepszej orientacji
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero (brak korelacji)')
    
    # Obliczamy i zaznaczamy średnią
    mean_weight = np.mean(weights)
    plt.axvline(mean_weight, color='green', linestyle=':', linewidth=2, label=f'Średnia ({mean_weight:.4f})')
    
    plt.legend()
    plt.tight_layout() # Dopasowuje wykres, aby nic nie było ucięte

    # --- Krok 5: Interpretacja dla użytkownika ---
    print("\n--- Interpretacja Wyników ---")
    print("Na wygenerowanym wykresie szukaj następujących cech (dowodów na emergencję):")
    print("1. WYSOKI SZCZYT WOKÓŁ ZERA: Większość par neuronów nie jest ze sobą skorelowana. Działają niezależnie.")
    print("2. CIĘŻKIE OGONY (HEAVY TAILS): Wartości daleko od zera (zwłaszcza dodatnie) występują znacznie częściej, niż w losowym szumie. To oznacza, że model 'odkrył' i stworzył grupy silnie powiązanych ze sobą neuronów (moduły), które współpracują przy konkretnych zadaniach. To jest właśnie emergencja struktury!")
    
    plt.show()

if __name__ == '__main__':
    main()