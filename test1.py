#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from multiprocessing import Pool, cpu_count
from mpmath import mp, pi as mp_pi, e as mp_e, sqrt as mp_sqrt

# =========================
# CONFIGURATION
# =========================
NUM_DIGITS = 1_000_000      # increase on HPC
BASES = [2, 8, 10, 16]
CONSTANTS = ['pi', 'e', 'root2', 'champernowne', 'random', 'periodic']
OUTPUT_DIR = "entropy_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# PLOTTING STYLE
# =========================
plt.rcParams.update({
    'figure.figsize': (9, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.4
})

def add_grids(ax):
    ax.grid(True, which='major', linewidth=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

# =========================
# DIGIT GENERATION (SAFE)
# =========================
def base_digits_from_real(x, base, n):
    mp.dps = int(n * np.log10(base)) + 50
    frac = x - int(x)
    digits = np.zeros(n, dtype=np.int16)

    for i in range(n):
        frac *= base
        d = int(frac)
        digits[i] = d
        frac -= d

    return digits

def get_digits(name, base, n):
    if name == 'pi':
        return base_digits_from_real(mp_pi, base, n)
    if name == 'e':
        return base_digits_from_real(mp_e, base, n)
    if name == 'root2':
        return base_digits_from_real(mp_sqrt(2), base, n)
    if name == 'champernowne':
        seq = []
        k = 1
        while len(seq) < n:
            s = np.base_repr(k, base)
            seq.extend([int(c, base=base) for c in s])
            k += 1
        return np.array(seq[:n], dtype=np.int16)
    if name == 'random':
        return np.random.randint(0, base, n, dtype=np.int16)
    if name == 'periodic':
        return np.tile(np.arange(base), n // base + 1)[:n]
    raise ValueError("Unknown constant")

# =========================
# ENTROPY DEFINITIONS
# =========================
def shannon_entropy(digits, base):
    counts = np.bincount(digits, minlength=base)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def hydrodynamic_entropy(digits):
    signal = digits - np.mean(digits)
    psi = np.fft.fft(signal)
    Ek = np.abs(psi)**2
    pk = Ek / Ek.sum()
    pk = pk[pk > 0]
    return -np.sum(pk * np.log2(pk))

# =========================
# ANALYSIS FUNCTION (PARALLEL UNIT)
# =========================
def analyze_case(args):
    constant, base = args
    print(f"[PID {os.getpid()}] {constant.upper()} | base {base}")

    outdir = os.path.join(OUTPUT_DIR, constant)
    os.makedirs(outdir, exist_ok=True)

    digits = get_digits(constant, base, NUM_DIGITS)

    # ---- Shannon entropy vs N
    Ns = np.unique(np.logspace(2, np.log10(NUM_DIGITS), 60, dtype=int))
    Hs = [shannon_entropy(digits[:n], base) for n in Ns]

    fig, ax = plt.subplots()
    ax.semilogx(Ns, Hs, lw=2)
    ax.axhline(np.log2(base), ls='--', color='k')
    ax.set_xlabel("N")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title(f"{constant.upper()} | Base {base}")
    add_grids(ax)
    plt.savefig(f"{outdir}/{constant}_base{base}_shannon.png")
    plt.close()

    # ---- Hydrodynamic entropy vs log10(N)
    Nh = Ns[Ns > 100]
    Sh = [hydrodynamic_entropy(digits[:n]) for n in Nh]

    fig, ax = plt.subplots()
    ax.plot(np.log10(Nh), Sh, lw=2)
    ax.set_xlabel("log₁₀(N)")
    ax.set_ylabel("Hydrodynamic entropy (bits)")
    ax.set_title(f"{constant.upper()} | Base {base}")
    add_grids(ax)
    plt.savefig(f"{outdir}/{constant}_base{base}_hydro.png")
    plt.close()

    # ---- Energy spectrum
    signal = digits - np.mean(digits)
    psi = np.fft.fft(signal)
    Ek = np.abs(psi[:len(psi)//2])**2
    k = np.arange(1, len(Ek) + 1)

    fig, ax = plt.subplots()
    ax.loglog(k, Ek, lw=1)
    ax.set_xlabel("Mode k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"{constant.upper()} | Base {base}")
    add_grids(ax)
    plt.savefig(f"{outdir}/{constant}_base{base}_energy.png")
    plt.close()

    return f"{constant} base {base} done"

# =========================
# MAIN (HPC SAFE)
# =========================
def main():
    cases = [(c, b) for c in CONSTANTS for b in BASES]

    nproc = min(cpu_count(), len(cases))
    print(f"Using {nproc} processes")

    with Pool(processes=nproc) as pool:
        pool.map(analyze_case, cases)

    print("\nAll analyses completed.")

if __name__ == "__main__":
    main()
