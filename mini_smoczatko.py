# mini_smoczatko.py
# Kompletny projekt implementacji i treningu modelu BDH-GPU

import torch
import torch.nn as nn
import torch.nn.functional as F
import requests  # Do pobierania danych
from tqdm import tqdm # Pasek postępu

# --- Parametry Modelu ---
# Możesz je zmniejszyć, jeśli trening na Twoim sprzęcie jest zbyt wolny
D = 128          # Wymiar wewnętrzny (małe "biurko robocze")
H = 4            # Liczba głowic uwagi
N = 4096         # Liczba neuronów (małe "archiwum koncepcyjne")
L = 4            # Liczba warstw
dropout = 0.1
vocab_size = 256 # Pracujemy na bajtach (UTF-8)
BLOCK_SIZE = 128 # Długość kontekstu (jak długie fragmenty tekstu analizujemy naraz)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5   # Liczba epok treningu
BATCH_SIZE = 32

# Sprawdzenie, czy dostępne jest GPU
# --- 1. Setup Device (CUDA or CPU) ---
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Implementacja RoPE (Rotary Positional Encoding) ---
# Kluczowy element nowoczesnych Transformerów, który dodaje informację o pozycji
def RoPE(x):
    shape = x.shape
    dim = shape[-1]
    seq_len = shape[-2]
    
    position = torch.arange(seq_len, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=x.device) * -(torch.log(torch.tensor(10000.0)) / dim))
    angles = position * div_term
    
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    x_rotated = torch.stack(
        [x_even * cos_angles - x_odd * sin_angles, 
         x_even * sin_angles + x_odd * cos_angles], 
        dim=-1
    )
    return x_rotated.flatten(-2)

# --- Architektura Modelu (z artykułu, z poprawkami) ---
class LinearAttention(nn.Module):
    def forward(self, Q, K, V):
        Qr = RoPE(Q)
        Kr = RoPE(K)
        
        scores = Qr @ Kr.transpose(-2, -1)
        mask = torch.tril(torch.ones_like(scores))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # W oryginalnym artykule jest uwaga liniowa, która unika softmaxu.
        # Dla prostoty i stabilności treningu użyjemy standardowego softmaxu.
        # Można to zamienić na (Qr @ Kr.transpose(-2, -1)).tril() dla ścisłej implementacji
        p_attn = F.softmax(scores, dim=-1)
        
        return p_attn @ V

class BDH_GPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, D)
        self.encoder = nn.Parameter(torch.randn(N, D) * 0.02)
        self.decoder_x = nn.Parameter(torch.randn(H, D // H, N // H) * 0.02) # POPRAWKA TUTAJ
        self.decoder_y = nn.Parameter(torch.randn(H, D // H, N // H) * 0.02) # I TUTAJ
        self.attn = LinearAttention()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.readout = nn.Linear(D, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        v_ast = self.wte(idx) # B, T, D

        for _ in range(L):
            v_ast_norm = self.ln1(v_ast)
            
            # Reshape do pracy z głowicami
            v_ast_reshaped = v_ast_norm.view(B, T, H, D // H).transpose(1, 2) # B, H, T, D//H

            # EKSPANSJA: z "biurka" D do "archiwum" N
            x = F.relu(torch.einsum('bhtd,hdi->bhti', v_ast_reshaped, self.decoder_x))

            # UWAGA w przestrzeni koncepcyjnej N
            a_ast = self.attn(Q=x, K=x, V=v_ast_reshaped) # B, H, T, D//H

            # MODULACJA I KOMPRESJA
            y_intermediate = F.relu(torch.einsum('bhtd,hdi->bhti', a_ast, self.decoder_y)) * x
            y = y_intermediate.transpose(1, 2).reshape(B, T, N)
            
            # KOMPRESJA: z powrotem z "archiwum" N na "biurko" D
            v_ast = v_ast + torch.einsum('btn,nd->btd', y, self.encoder)
            v_ast = v_ast + self.ln2(v_ast) # Drugie połączenie rezydualne i normalizacja

        logits = self.readout(v_ast)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Przygotowanie Danych ---
def get_data():
    url = "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        with open("pan-tadeusz.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Pobrano 'Pan Tadeusz'")
    except requests.exceptions.RequestException as e:
        print(f"Błąd pobierania danych: {e}")
        return None
    return text

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- Główna Część Skryptu ---
if __name__ == '__main__':
    # 1. Wczytaj i przygotuj dane
    raw_text = get_data()
    if not raw_text:
        exit()
        
    # Prosta tokenizacja na poziomie bajtów
    data = torch.tensor(bytearray(raw_text, 'utf-8'), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 2. Stwórz model i przenieś na GPU
    model = BDH_GPU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Pętla treningowa
    print(f"Rozpoczynam trening modelu z {sum(p.numel() for p in model.parameters())/1e6:.2f}M parametrów.")
    for epoch in range(NUM_EPOCHS):
        model.train()
        pbar = tqdm(range(1000)) # 1000 kroków na epokę
        for i in pbar:
            xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
            
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoka {epoch+1}/{NUM_EPOCHS} | Strata: {loss.item():.4f}")

    # 4. Zapisz wytrenowany model
    torch.save(model.state_dict(), 'mini_smoczatko.pth')
    print("Zapisano model do pliku mini_smoczatko.pth")

    # 5. Generowanie tekstu
    print("\n--- Generowanie tekstu po treningu ---")
    model.eval()
    context = torch.tensor(bytearray("Litwo, Ojczyzno moja!", 'utf-8'), dtype=torch.long, device=device).unsqueeze(0)
    generated_bytes = model.generate(context, max_new_tokens=500)[0].tolist()
    
    try:
        print(bytearray(generated_bytes).decode('utf-8', errors='ignore'))
    except Exception as e:
        print(f"Błąd dekodowania: {e}")