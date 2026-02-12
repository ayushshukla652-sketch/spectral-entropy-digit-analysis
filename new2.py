import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import os

# ============================================================
# SETTINGS
# ============================================================

N_total = 2_000_000
num_samples = 100
SAVE_DIR = "extended_plots"

os.makedirs(SAVE_DIR, exist_ok=True)

np.random.seed(0)

# ============================================================
# GENERATORS
# ============================================================

# Logistic map (r = 4)
def generate_logistic(N):
    x = 0.123456
    digits = np.empty(N, dtype=np.uint8)
    for i in range(N):
        x = 4 * x * (1 - x)
        digits[i] = int(x * 10) % 10
    return digits

# LCG
def generate_lcg(N, a=1664525, c=1013904223, m=2**32):
    x = 1
    digits = np.empty(N, dtype=np.uint8)
    for i in range(N):
        x = (a * x + c) % m
        digits[i] = x % 10
    return digits

# Thue-Morse
def generate_thue_morse(N):
    digits = np.empty(N, dtype=np.uint8)
    for i in range(N):
        digits[i] = (bin(i).count("1") % 2) * 5
    return digits

# De Bruijn (base 10, order 7 gives 10^7 length)
def debruijn(k, n):
    a = [0] * (k * n)
    sequence = []
    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p+1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1,1)
    return np.array(sequence, dtype=np.uint8)

# ============================================================
# SPECTRAL ENTROPY
# ============================================================

def spectral_entropy(signal):
    signal = signal.astype(np.float64)
    signal -= np.mean(signal)
    fft_vals = rfft(signal)
    power = np.abs(fft_vals)**2
    p = power / np.sum(power)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def compute_scaling(digits):
    N_vals = np.unique(
        np.logspace(3, np.log10(N_total), num_samples).astype(int)
    )
    S_vals = []
    for N in N_vals:
        S_vals.append(spectral_entropy(digits[:N]))
    return np.log10(N_vals), np.array(S_vals)

# ============================================================
# GENERATE ALL SEQUENCES
# ============================================================

print("Generating sequences...")

logistic_digits = generate_logistic(N_total)
lcg_digits = generate_lcg(N_total)
thue_digits = generate_thue_morse(N_total)
db_digits = debruijn(10, 7)  # length = 10^7 exactly

sequences = {
    "Logistic": logistic_digits,
    "LCG": lcg_digits,
    "Thue-Morse": thue_digits,
    "DeBruijn": db_digits
}

# ============================================================
# PLOT SPECTRAL ENTROPY FOR EACH
# ============================================================

for name, digits in sequences.items():

    print(f"Processing {name}...")

    logN, S_vals = compute_scaling(digits)

    plt.figure(figsize=(7,6))

    if name == "DeBruijn":
        # Quadratic fit
        coeffs = np.polyfit(logN, S_vals, 2)
        a, b, c = coeffs

        x = np.linspace(logN.min(), logN.max(), 400)
        plt.plot(logN, S_vals, alpha=0.4)
        plt.plot(x, a*x**2 + b*x + c, linewidth=3)

        eq = r"$\alpha_{\rm eff}(N)=%.3f\,\log_{10}N %+ .3f$"%(2*a, b)
        plt.text(0.05, 0.95, eq,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))

    else:
        coeffs = np.polyfit(logN, S_vals, 1)
        alpha, beta = coeffs
        plt.plot(logN, S_vals, linewidth=2,
                 label=r"$\alpha \approx %.3f$"%alpha)
        plt.legend()

    # theoretical reference
    xline = np.linspace(logN.min(), logN.max(), 200)
    plt.plot(xline, 3.322*xline - 1, 'k--', label="Theory α=3.322")

    plt.xlabel("log10(N)")
    plt.ylabel("Spectral Entropy $S_H$")
    plt.title(f"{name} Spectral Entropy Scaling")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{SAVE_DIR}/{name}_spectral_entropy.png", dpi=300)
    plt.close()

print(f"\nAll plots saved inside folder: {SAVE_DIR}")
