import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

# -------------------------------------------------
# 1. Generate De Bruijn sequence (base 10, order k)
# -------------------------------------------------
def de_bruijn(k, n):
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
    return np.array(sequence)

digits = de_bruijn(10, 7)

# -------------------------------------------------
# 2. Spectral Entropy
# -------------------------------------------------
def spectral_entropy(signal):
    N = len(signal)
    signal = signal - np.mean(signal)
    spectrum = fft(signal)
    power = np.abs(spectrum[:N//2])**2
    power /= np.sum(power)
    return -np.sum(power * np.log2(power + 1e-15))

# -------------------------------------------------
# 3. Compute entropy vs N
# -------------------------------------------------
N_values = np.logspace(2, np.log10(len(digits)), 500).astype(int)
N_values = np.unique(N_values)

S_values = []
logN = []

for N in N_values:
    S = spectral_entropy(digits[:N])
    S_values.append(S)
    logN.append(np.log10(N))

S_values = np.array(S_values)
logN = np.array(logN)

# -------------------------------------------------
# 4. Quadratic Fit
# -------------------------------------------------
coeffs = np.polyfit(logN, S_values, 2)
a, b, c = coeffs

x_fit = np.linspace(logN.min(), logN.max(), 500)
y_fit = a*x_fit**2 + b*x_fit + c

# -------------------------------------------------
# 5. Plot
# -------------------------------------------------
plt.figure(figsize=(7,5))

plt.scatter(logN, S_values, s=15, alpha=0.7, label="De Bruijn Data")
plt.plot(x_fit, y_fit, 'r', linewidth=2,
         label="Quadratic Best Fit")

# ---- Display derivative α(x) ----
alpha_text = r"$\alpha(x) = %.4f\,x %.4f$" % (2*a, b)

plt.text(0.05, 0.95, alpha_text,
         transform=plt.gca().transAxes,
         fontsize=11,
         verticalalignment='top')

plt.xlabel("log10(N)")
plt.ylabel("Spectral Entropy $S_H$")
plt.title("De Bruijn Spectral Entropy (Nonlinear Scaling)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
