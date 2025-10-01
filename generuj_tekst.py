# generuj_tekst.py
# Skrypt do generowania tekstu przy użyciu wytrenowanego modelu "Mini-Smoczątko".

import torch
import sys

# Importujemy klasę modelu i parametry, aby poprawnie go zbudować
try:
    from mini_smoczatko import BDH_GPU, BLOCK_SIZE, device
except ImportError:
    print("Błąd: Upewnij się, że plik 'mini_smoczatko.py' jest w tym samym folderze.")
    sys.exit(1)

MODEL_PATH = 'mini_smoczatko.pth'
OUTPUT_FILE = 'wygenerowany_tekst.txt'

def generate():
    # --- Krok 1: Wczytaj wytrenowany model ---
    try:
        model = BDH_GPU().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Przełącz model w tryb ewaluacji (ważne!)
        print(f"Pomyślnie wczytano model z pliku '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku modelu '{MODEL_PATH}'.")
        print("Najpierw uruchom 'mini_smoczatko.py', aby wytrenować i zapisać model.")
        return
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
        return

    # --- Krok 2: Przygotuj zdanie początkowe ---
    prompt = "Litwo, Ojczyzno moja! ty jesteś jak zdrowie."
    print(f"\nZdanie początkowe: '{prompt}'")
    
    # Przekształć tekst na tokeny (bajty)
    context = torch.tensor(bytearray(prompt, 'utf-8'), dtype=torch.long, device=device).unsqueeze(0)

    # --- Krok 3: Generuj nowy tekst ---
    print("\nGenerowanie nowego tekstu (1000 znaków)...")
    with torch.no_grad(): # Wyłącz obliczanie gradientów, aby przyspieszyć i oszczędzić pamięć
        generated_bytes = model.generate(context, max_new_tokens=1000)[0].tolist()
    
    # --- Krok 4: Zdekoduj i wyświetl wynik ---
    try:
        generated_text = bytearray(generated_bytes).decode('utf-8', errors='replace')
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
    except Exception as e:
        print(f"Błąd podczas dekodowania wygenerowanych bajtów: {e}")
        return
        
    # --- Krok 5: Zapisz wynik do pliku ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nPełny wygenerowany tekst został zapisany do pliku: '{OUTPUT_FILE}'")
    except IOError as e:
        print(f"Błąd podczas zapisywania do pliku: {e}")

if __name__ == '__main__':
    generate()
