import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import gc

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

MAX_POWER = 7       # 10^7 safe start (increase to 8 later)
POINTS = 18
SEQUENCE_TYPE = "debruijn"  # options below

# OPTIONS:
# "debruijn"
# "thue"
# "logistic"
# "lcg"

# ------------------------------------------------
# 1️⃣ De Bruijn sequence (base 10, order k)
# ------------------------------------------------

def de_bruijn(k, alphabet=10):
    """Generate de Bruijn sequence B(10, k)"""
    a = [0] * (alphabet * k)
    sequence = []

    def db(t, p):
        if t > k:
            if k % p == 0:
                sequence.extend(a[1:p+1])
        else:
            a[t] = a[t-p]
            db(t+1, p)
            for j in range(a[t-p]+1, alphabet):
                a[t] = j
                db(t+1, t)

    db(1,1)
    return np.array(sequence, dtype=np.int8)

# ------------------------------------------------
# 2️⃣ Thue-Morse (binary → mapped to digits 0–9)
# ------------------------------------------------

def thue_morse(N):
    seq = np.zeros(N, dtype=np.int8)
    for i in range(N):
        seq[i] = bin(i).count("1") % 2
    # map 0/1 → 0–9 spread
    return (seq * 9).astype(np.int8)

# ------------------------------------------------
# 3️⃣ Logistic Map
# ------------------------------------------------

def logistic_sequence(N, r=4.0, x0=0.123456):
    x = x0
    seq = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = r * x * (1 - x)
        seq[i] = int(x * 10) % 10
    return seq

# ------------------------------------------------
# 4️⃣ LCG PRNG
# ------------------------------------------------

def lcg_sequence(N, seed=123456):
    a = 1664525
    c = 1013904223
    m = 2**32
    x = seed
    seq = np.zeros(N, dtype=np.int8)
    for i in range(N):
        x = (a*x + c) % m
        seq[i] = (x % 10)
    return seq

# ------------------------------------------------
# Shannon Entropy
# ------------------------------------------------

def shannon_entropy(digits):
    counts = np.bincount(digits, minlength=10)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

# ------------------------------------------------
# Spectral Entropy
# ------------------------------------------------

def spectral_entropy(digits):
    signal = digits.astype(np.float32)
    signal -= np.mean(signal)

    fft_vals = rfft(signal)
    power = np.abs(fft_vals)**2

    power = power[1:]
    total_energy = np.sum(power)

    if total_energy == 0:
        return 0.0

    pk = power / total_energy
    SH = -np.sum(pk * np.log2(pk + 1e-15))

    del signal, power, pk
    gc.collect()

    return SH

# ------------------------------------------------
# Sequence Selector
# ------------------------------------------------

def generate_sequence(N):
    if SEQUENCE_TYPE == "debruijn":
        k = int(np.log(N)/np.log(10))
        seq = de_bruijn(k)
        return np.tile(seq, int(np.ceil(N/len(seq))))[:N]

    elif SEQUENCE_TYPE == "thue":
        return thue_morse(N)

    elif SEQUENCE_TYPE == "logistic":
        return logistic_sequence(N)

    elif SEQUENCE_TYPE == "lcg":
        return lcg_sequence(N)

    else:
        raise ValueError("Unknown sequence type")

# ------------------------------------------------
# Main Computation
# ------------------------------------------------

N_values = np.logspace(3, MAX_POWER, POINTS, dtype=int)

H_values = []
SH_values = []

for N in N_values:
    print(f"\nComputing N = {N}")
    digits = generate_sequence(N)

    H = shannon_entropy(digits)
    SH = spectral_entropy(digits)

    H_values.append(H)
    SH_values.append(SH)

    del digits
    gc.collect()

H_values = np.array(H_values)
SH_values = np.array(SH_values)

# ------------------------------------------------
# Fit Scaling
# ------------------------------------------------

logN = np.log10(N_values)
alpha, beta = np.polyfit(logN, SH_values, 1)

print("\nFitted Scaling:")
print(f"alpha = {alpha:.4f}")

# ------------------------------------------------
# Plot Shannon
# ------------------------------------------------

plt.figure(figsize=(6,5))
plt.plot(logN, H_values, 'o-', linewidth=2)
plt.axhline(np.log2(10), linestyle='--')
plt.xlabel("log10(N)")
plt.ylabel("Shannon Entropy")
plt.title(f"Shannon Entropy – {SEQUENCE_TYPE}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"shannon_{SEQUENCE_TYPE}.pdf", dpi=300)

# ------------------------------------------------
# Plot Spectral
# ------------------------------------------------

plt.figure(figsize=(6,5))
plt.plot(logN, SH_values, 'o-', linewidth=2, label="Data")
plt.plot(logN, alpha*logN + beta, '--', label=f"α={alpha:.2f}")
plt.xlabel("log10(N)")
plt.ylabel("Spectral Entropy")
plt.title(f"Spectral Entropy – {SEQUENCE_TYPE}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"spectral_{SEQUENCE_TYPE}.pdf", dpi=300)

plt.show()
