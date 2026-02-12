#!/usr/bin/env python3
"""
Spectral Entropy Analysis of Mathematical Constants
=====================================================
Computes Shannon entropy and spectral (hydrodynamic) entropy for decimal
digit sequences of various mathematical constants, fits scaling laws,
and produces summary tables + plots.

Author: Ayush Shukla (exploratory code)

USAGE:
    python spectral_analysis.py

Adjust N_DIGITS and SLOW_DIGITS below based on your machine.
"""

import numpy as np
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpmath import mp, mpf, pi, e, euler, phi, zeta, ln, sqrt, catalan, khinchin
import time
import os
import math
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — ADJUST THESE BASED ON YOUR MACHINE
# ============================================================
# PERFORMANCE NOTES:
#   - pi, e, sqrt2, phi, ln2, zeta3, catalan: Fast (~seconds for 1M digits)
#   - euler_gamma: SLOW (~25s for 50k, ~10min for 200k, hours for 1M).
#   - khinchin: EXTREMELY SLOW. Recommend 50k max or skip entirely.
#   - copeland_erdos: Needs large prime sieve. Fast up to 1M.
#   - champernowne: Fast (string concatenation).
#
# For a first exploratory run, use N_DIGITS = 200_000.
# For paper-quality results, use N_DIGITS = 1_500_000 but
# keep SLOW_DIGITS small for gamma and Khinchin.
# ============================================================

N_DIGITS = 200_000            # Max digits for fast constants
SLOW_DIGITS = 50_000          # Max digits for slow constants (gamma, khinchin)
N_POINTS = 600                # Number of log-spaced sample points
FIT_LOWER = 1000              # Lower bound for fitting range
OUTPUT_DIR = "spectral_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants that are very slow to compute at high precision
SLOW_CONSTANTS = {"euler_gamma", "khinchin"}

# Set mpmath precision
mp.dps = N_DIGITS + 100

# ============================================================
# DIGIT EXTRACTION FUNCTIONS
# ============================================================

def extract_decimal_digits(value, n_digits):
    """Extract first n_digits decimal digits after the decimal point."""
    mp.dps = n_digits + 50
    val = mpf(value)
    val_str = mp.nstr(val, n_digits + 20, strip_zeros=False)
    if '.' in val_str:
        _, frac_part = val_str.split('.')
    else:
        frac_part = ''
    while len(frac_part) < n_digits:
        frac_part += '0'
    digits = [int(c) for c in frac_part[:n_digits]]
    return np.array(digits, dtype=np.int8)


def compute_constant(name, n_digits):
    """Compute digits for a named constant. Returns numpy array of digits."""
    mp.dps = n_digits + 100
    print(f"  Computing {name} ({n_digits:,} digits)...", end=" ", flush=True)
    t0 = time.time()

    if name == "pi":
        val = pi
    elif name == "e":
        val = e
    elif name == "sqrt2":
        val = sqrt(2)
    elif name == "sqrt3":
        val = sqrt(3)
    elif name == "sqrt5":
        val = sqrt(5)
    elif name == "phi":
        val = phi
    elif name == "euler_gamma":
        val = euler
    elif name == "zeta3":
        val = zeta(3)
    elif name == "ln2":
        val = ln(2)
    elif name == "ln10":
        val = ln(10)
    elif name == "catalan":
        val = catalan
    elif name == "khinchin":
        val = khinchin
    elif name == "pi_squared":
        val = pi ** 2
    elif name == "e_pi":
        val = e ** pi
    elif name == "sqrt2_plus_sqrt3":
        val = sqrt(2) + sqrt(3)
    elif name == "champernowne":
        digits = generate_champernowne(n_digits)
        print(f"done ({time.time() - t0:.1f}s)")
        return digits
    elif name == "copeland_erdos":
        digits = generate_copeland_erdos(n_digits)
        print(f"done ({time.time() - t0:.1f}s)")
        return digits
    elif name == "liouville":
        digits = generate_liouville_number(n_digits)
        print(f"done ({time.time() - t0:.1f}s)")
        return digits
    elif name == "periodic":
        digits = np.array([i % 10 for i in range(n_digits)], dtype=np.int8)
        print(f"done ({time.time() - t0:.1f}s)")
        return digits
    elif name == "random":
        digits = np.random.randint(0, 10, size=n_digits).astype(np.int8)
        print(f"done ({time.time() - t0:.1f}s)")
        return digits
    else:
        raise ValueError(f"Unknown constant: {name}")

    digits = extract_decimal_digits(val, n_digits)
    print(f"done ({time.time() - t0:.1f}s)")
    return digits


def generate_champernowne(n_digits):
    """Generate Champernowne constant C_10 digits."""
    digits = []
    num = 1
    while len(digits) < n_digits:
        digits.extend([int(c) for c in str(num)])
        num += 1
    return np.array(digits[:n_digits], dtype=np.int8)


def sieve_primes(limit):
    """Simple sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]


def generate_copeland_erdos(n_digits):
    """Generate Copeland-Erdos constant (concatenation of primes)."""
    limit = max(10 * n_digits, 1_000_000)
    primes = sieve_primes(limit)
    digits = []
    for p in primes:
        if len(digits) >= n_digits:
            break
        digits.extend([int(c) for c in str(p)])
    if len(digits) < n_digits:
        print(f"\n  WARNING: Only got {len(digits)} digits for Copeland-Erdos")
    return np.array(digits[:n_digits], dtype=np.int8)


def generate_liouville_number(n_digits):
    """
    Generate Liouville number L = sum_{k=1}^{inf} 10^{-k!}
    = 0.110001000000000000000001000...
    """
    digits = np.zeros(n_digits, dtype=np.int8)
    k = 1
    while True:
        pos = math.factorial(k) - 1  # 0-indexed
        if pos >= n_digits:
            break
        digits[pos] = 1
        k += 1
    return digits


# ============================================================
# ENTROPY COMPUTATION FUNCTIONS
# ============================================================

def shannon_entropy(digits):
    """Compute Shannon entropy of digit sequence in bits."""
    N = len(digits)
    counts = np.bincount(digits, minlength=10)
    probs = counts / N
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def spectral_entropy(digits):
    """Compute spectral (hydrodynamic) entropy of digit sequence."""
    N = len(digits)
    x = digits.astype(np.float64)
    x = x - np.mean(x)
    psi = np.fft.fft(x)
    M = N // 2
    Ek = 0.5 * np.abs(psi[1:M + 1]) ** 2
    total_energy = np.sum(Ek)
    if total_energy == 0:
        return 0.0
    pk = Ek / total_energy
    pk = pk[pk > 0]
    return -np.sum(pk * np.log2(pk))


def analyze_sequence(digits, n_points=None):
    """
    Compute Shannon and spectral entropy at logarithmically spaced N values.
    """
    if n_points is None:
        n_points = N_POINTS
    max_n = len(digits)
    log_n_values = np.linspace(1, np.log10(max_n), n_points)
    N_values = np.unique(np.round(10 ** log_n_values).astype(int))
    N_values = N_values[(N_values >= 10) & (N_values <= max_n)]

    H_list, SH_list, N_list = [], [], []
    for N in N_values:
        d = digits[:N]
        H_list.append(shannon_entropy(d))
        SH_list.append(spectral_entropy(d))
        N_list.append(N)

    return np.array(N_list), np.array(H_list), np.array(SH_list)


def fit_spectral_scaling(N_arr, SH_arr, fit_lower=None, fit_upper=None):
    """
    Fit SH = alpha * log10(N) + beta over specified range.
    Returns alpha, beta, R^2, alpha_95ci.
    """
    if fit_lower is None:
        fit_lower = FIT_LOWER
    if fit_upper is None:
        fit_upper = N_arr[-1]
    mask = (N_arr >= fit_lower) & (N_arr <= fit_upper)
    if np.sum(mask) < 10:
        return 0.0, 0.0, 0.0, 0.0

    log_N = np.log10(N_arr[mask])
    SH = SH_arr[mask]
    result = linregress(log_N, SH)
    alpha = result.slope
    beta = result.intercept
    r_squared = result.rvalue ** 2
    alpha_95ci = 1.96 * result.stderr
    return alpha, beta, r_squared, alpha_95ci


def fit_sensitivity(N_arr, SH_arr):
    """Test sensitivity of alpha to fitting range lower bound."""
    lower_bounds = [100, 200, 500, 1000, 2000, 5000, 10000]
    results = []
    for lb in lower_bounds:
        alpha, beta, r2, ci = fit_spectral_scaling(N_arr, SH_arr, fit_lower=lb)
        results.append((lb, alpha, r2, ci))
    return results


# ============================================================
# LEMPEL-ZIV COMPLEXITY
# ============================================================

def lempel_ziv_complexity(sequence):
    """Compute Lempel-Ziv complexity (LZ76) of a sequence."""
    s = ''.join(str(d) for d in sequence)
    n = len(s)
    if n == 0:
        return 0, 0

    c = 1
    l = 1
    k = 1

    while l + k <= n:
        if s[l:l + k] in s[0:l + k - 1]:
            k += 1
        else:
            c += 1
            l += k
            k = 1

    if n > 1:
        b = 10
        normalized = c / (n / np.log(n) * np.log(b))
    else:
        normalized = 0

    return c, normalized


# ============================================================
# CONSTANTS DATABASE
# ============================================================

CONSTANTS = {
    "pi": {
        "display": "pi",
        "normal": "Conjectured",
        "type": "Transcendental",
        "description": "Archimedes' constant",
    },
    "e": {
        "display": "e",
        "normal": "Conjectured",
        "type": "Transcendental",
        "description": "Euler's number",
    },
    "sqrt2": {
        "display": "sqrt2",
        "normal": "Open",
        "type": "Algebraic",
        "description": "Square root of 2",
    },
    "phi": {
        "display": "phi",
        "normal": "Open",
        "type": "Algebraic",
        "description": "Golden ratio (1+sqrt5)/2",
    },
    "sqrt3": {
        "display": "sqrt3",
        "normal": "Open",
        "type": "Algebraic",
        "description": "Square root of 3",
    },
    "sqrt5": {
        "display": "sqrt5",
        "normal": "Open",
        "type": "Algebraic",
        "description": "Square root of 5",
    },
    "euler_gamma": {
        "display": "gamma",
        "normal": "Open",
        "type": "Unknown (conj. transcendental)",
        "description": "Euler-Mascheroni constant",
    },
    "zeta3": {
        "display": "zeta(3)",
        "normal": "Open",
        "type": "Irrational (transcendence open)",
        "description": "Apery's constant",
    },
    "ln2": {
        "display": "ln(2)",
        "normal": "Open",
        "type": "Transcendental (has BBP formula)",
        "description": "Natural log of 2",
    },
    "catalan": {
        "display": "G",
        "normal": "Open",
        "type": "Unknown (irrationality unproven)",
        "description": "Catalan's constant",
    },
    "khinchin": {
        "display": "K0",
        "normal": "Open",
        "type": "Unknown",
        "description": "Khinchin's constant",
    },
    "ln10": {
        "display": "ln(10)",
        "normal": "Open",
        "type": "Transcendental",
        "description": "Natural log of 10",
    },
    "e_pi": {
        "display": "e^pi",
        "normal": "Open",
        "type": "Transcendental",
        "description": "Gelfond's constant",
    },
    "champernowne": {
        "display": "C10",
        "normal": "YES (proven)",
        "type": "Transcendental (constructed)",
        "description": "Champernowne constant",
    },
    "copeland_erdos": {
        "display": "C_CE",
        "normal": "YES (proven)",
        "type": "Irrational (concat. of primes)",
        "description": "Copeland-Erdos constant",
    },
    "liouville": {
        "display": "L",
        "normal": "NO (non-normal)",
        "type": "Transcendental",
        "description": "Liouville number",
    },
    "periodic": {
        "display": "Periodic",
        "normal": "NO (non-normal)",
        "type": "Rational",
        "description": "(0,1,2,...,9) repeating",
    },
    "random": {
        "display": "Random",
        "normal": "YES (by construction)",
        "type": "Uniform iid",
        "description": "Pseudo-random digits",
    },
}


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    alpha_theory = np.log(10) / np.log(2)

    print("=" * 70)
    print("SPECTRAL ENTROPY ANALYSIS OF MATHEMATICAL CONSTANTS")
    print("=" * 70)
    print(f"Fast constants: {N_DIGITS:,} digits")
    print(f"Slow constants: {SLOW_DIGITS:,} digits")
    print(f"Fitting lower bound: N >= {FIT_LOWER}")
    print(f"Theoretical alpha = ln(10)/ln(2) = {alpha_theory:.4f}")
    print()

    # ---- Phase 1: Compute digits ----
    print("PHASE 1: Computing digits")
    print("-" * 40)

    all_digits = {}
    for name in CONSTANTS:
        n = SLOW_DIGITS if name in SLOW_CONSTANTS else N_DIGITS
        try:
            digits = compute_constant(name, n)
            all_digits[name] = digits
        except Exception as ex:
            print(f"  ERROR computing {name}: {ex}")

    print(f"\nSuccessfully computed {len(all_digits)}/{len(CONSTANTS)} constants.\n")

    # ---- Phase 2: Entropy analysis ----
    print("PHASE 2: Computing entropies")
    print("-" * 40)

    all_results = {}
    for name, digits in all_digits.items():
        n_dig = len(digits)
        print(f"  Analyzing {CONSTANTS[name]['display']} ({n_dig:,} digits)...", end=" ", flush=True)
        t0 = time.time()

        N_arr, H_arr, SH_arr = analyze_sequence(digits)
        alpha, beta, r2, ci95 = fit_spectral_scaling(N_arr, SH_arr)
        sensitivity = fit_sensitivity(N_arr, SH_arr)

        # Lempel-Ziv on subsample (LZ is slow for large N)
        lz_n = min(100_000, len(digits))
        lz_c, lz_norm = lempel_ziv_complexity(digits[:lz_n])

        H_max = shannon_entropy(digits)

        all_results[name] = {
            "N": N_arr,
            "H": H_arr,
            "SH": SH_arr,
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "ci95": ci95,
            "sensitivity": sensitivity,
            "lz_count": lz_c,
            "lz_norm": lz_norm,
            "H_final": H_max,
            "n_digits": n_dig,
        }

        dt = time.time() - t0
        print(f"alpha={alpha:.3f} +/-{ci95:.3f}, R2={r2:.4f} ({dt:.1f}s)")

    # Sort by alpha descending
    sorted_names = sorted(
        all_results.keys(),
        key=lambda x: all_results[x]['alpha'],
        reverse=True,
    )

    # ---- Phase 3: Summary table ----
    print("\n" + "=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    header = (
        f"{'Constant':<18} {'Display':<10} {'Normal?':<25} {'Type':<35} "
        f"{'alpha':>6} {'95%CI':>8} {'R2':>7} {'H_final':>8} {'LZ_norm':>8} {'N_dig':>10}"
    )
    print(header)
    print("-" * 130)

    print(
        f"{'Theory':<18} {'--':<10} {'--':<25} {'--':<35} "
        f"{alpha_theory:>6.3f} {'--':>8} {'--':>7} {'3.322':>8} {'--':>8} {'--':>10}"
    )
    print("-" * 130)

    for name in sorted_names:
        r = all_results[name]
        c = CONSTANTS[name]
        print(
            f"{name:<18} {c['display']:<10} {c['normal']:<25} {c['type']:<35} "
            f"{r['alpha']:>6.3f} {r['ci95']:>7.3f} {r['r2']:>7.4f} "
            f"{r['H_final']:>8.4f} {r['lz_norm']:>8.4f} {r['n_digits']:>10,}"
        )

    # ---- Phase 4: Sensitivity analysis ----
    print("\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS: alpha vs. lower fitting bound")
    print("=" * 90)

    key_constants = [
        "pi", "e", "sqrt2", "phi", "euler_gamma", "zeta3", "ln2",
        "copeland_erdos", "champernowne", "liouville", "random",
    ]
    key_constants = [k for k in key_constants if k in all_results]

    sens_bounds = [100, 500, 1000, 5000, 10000]
    header2 = f"{'Constant':<18}"
    for lb in sens_bounds:
        header2 += f"  {'N>=' + str(lb):>10}"
    print(header2)
    print("-" * 90)

    for name in key_constants:
        sens = all_results[name]['sensitivity']
        line = f"{CONSTANTS[name]['display']:<18}"
        for lb, alpha_s, r2_s, ci_s in sens:
            if lb in sens_bounds:
                line += f"  {alpha_s:>10.3f}"
        print(line)

    # ---- Phase 5: Classification matrix ----
    print("\n" + "=" * 95)
    print("CLASSIFICATION MATRIX")
    print("=" * 95)
    print(
        f"{'':>25} | {'alpha >= 3.0 (high)':>25} | "
        f"{'1 < alpha < 3 (med)':>25} | {'alpha < 1 (low)':>15}"
    )
    print("-" * 95)

    categories = {}
    for name in sorted_names:
        r = all_results[name]
        c = CONSTANTS[name]
        norm = c['normal']

        if "YES" in norm:
            cat = "Normal (proven)"
        elif "Conjectured" in norm:
            cat = "Normal (conj.)"
        elif "Open" in norm:
            cat = "Normality open"
        else:
            cat = "Non-normal"

        if r['alpha'] >= 3.0:
            col = "high"
        elif r['alpha'] >= 1.0:
            col = "medium"
        else:
            col = "low"

        categories.setdefault(cat, []).append((name, c['display'], r['alpha'], col))

    for cat_name in ["Normal (proven)", "Normal (conj.)", "Normality open", "Non-normal"]:
        entries = categories.get(cat_name, [])
        high = [f"{d}({a:.2f})" for _, d, a, c in entries if c == "high"]
        med = [f"{d}({a:.2f})" for _, d, a, c in entries if c == "medium"]
        low = [f"{d}({a:.2f})" for _, d, a, c in entries if c == "low"]
        print(
            f"{cat_name:>25} | {', '.join(high) if high else '--':>25} | "
            f"{', '.join(med) if med else '--':>25} | "
            f"{', '.join(low) if low else '--':>15}"
        )

    # ---- Phase 6: Plots ----
    print("\n\nPHASE 6: Generating plots...")

    # --- Plot 1: All spectral entropies ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    for name in sorted_names:
        r = all_results[name]
        c = CONSTANTS[name]
        a = r['alpha']
        label = f"{c['display']} (alpha={a:.2f})"
        ls = '-' if a >= 3.0 else ('--' if a >= 1.0 else ':')
        lw = 1.5 if a >= 3.0 else 2.0
        ax.plot(r['N'], r['SH'], ls, linewidth=lw, label=label, alpha=0.8)

    n_max = max(r['n_digits'] for r in all_results.values())
    N_theory = np.logspace(1, np.log10(n_max), 100)
    SH_theory = alpha_theory * np.log10(N_theory) - 1
    ax.plot(
        N_theory, SH_theory, 'k--', linewidth=1, alpha=0.5,
        label=f'Theory (alpha={alpha_theory:.2f})',
    )

    ax.set_xscale('log')
    ax.set_xlabel('Number of digits (N)', fontsize=13)
    ax.set_ylabel('Spectral Entropy S_H (bits)', fontsize=13)
    ax.set_title('Spectral Entropy Scaling: All Constants', fontsize=15)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/all_spectral_entropy.png", dpi=200)
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/all_spectral_entropy.png")

    # --- Plot 2: Alpha bar chart ---
    fig, ax = plt.subplots(figsize=(14, 8))
    alphas = [all_results[n]['alpha'] for n in sorted_names]
    cis = [all_results[n]['ci95'] for n in sorted_names]
    displays = [CONSTANTS[n]['display'] for n in sorted_names]

    bar_colors = []
    for n in sorted_names:
        a = all_results[n]['alpha']
        bar_colors.append('#2196F3' if a >= 3.0 else ('#FF9800' if a >= 1.0 else '#F44336'))

    ax.barh(
        range(len(sorted_names)), alphas, xerr=cis, color=bar_colors,
        edgecolor='black', linewidth=0.5, capsize=3, alpha=0.85,
    )
    ax.axvline(
        x=alpha_theory, color='black', linestyle='--', linewidth=1.5,
        label=f'Theory: alpha = {alpha_theory:.3f}',
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(
        [f"{d}  ({CONSTANTS[n]['normal']})" for n, d in zip(sorted_names, displays)],
        fontsize=9,
    )
    ax.set_xlabel('Spectral Entropy Scaling Exponent alpha', fontsize=13)
    ax.set_title('Spectral Entropy Scaling alpha for All Constants', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alpha_barchart.png", dpi=200)
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/alpha_barchart.png")

    # --- Plot 3: Individual spectral entropy grid ---
    n_constants = len(sorted_names)
    ncols = 4
    nrows = max(1, (n_constants + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten()

    for i, name in enumerate(sorted_names):
        ax = axes_flat[i]
        r = all_results[name]
        c = CONSTANTS[name]
        ax.plot(r['N'], r['SH'], 'b-', linewidth=0.8, alpha=0.7)
        log_N = np.log10(r['N'])
        fit_line = r['alpha'] * log_N + r['beta']
        ax.plot(r['N'], fit_line, 'r--', linewidth=1.5, alpha=0.8)
        ax.set_xscale('log')
        ax.set_title(f"{c['display']}  alpha={r['alpha']:.3f}  R2={r['r2']:.4f}", fontsize=11)
        ax.set_xlabel('N', fontsize=9)
        ax.set_ylabel('S_H', fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle('Spectral Entropy: Individual Constants', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/individual_spectral_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/individual_spectral_grid.png")

    # --- Plot 4: Shannon entropy grid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = axes.flatten()

    for i, name in enumerate(sorted_names):
        ax = axes_flat[i]
        r = all_results[name]
        c = CONSTANTS[name]
        ax.plot(r['N'], r['H'], 'g-', linewidth=0.8, alpha=0.7)
        ax.axhline(y=np.log2(10), color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_title(f"{c['display']}  H={r['H_final']:.4f}", fontsize=11)
        ax.set_xlabel('N', fontsize=9)
        ax.set_ylabel('Shannon H', fontsize=9)
        ax.set_ylim([0, 3.5])
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle('Shannon Entropy: Individual Constants', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/individual_shannon_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/individual_shannon_grid.png")

    # --- Plot 5: Alpha vs Lempel-Ziv scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in sorted_names:
        r = all_results[name]
        c = CONSTANTS[name]
        ax.scatter(r['alpha'], r['lz_norm'], s=100, zorder=5)
        ax.annotate(
            c['display'], (r['alpha'], r['lz_norm']),
            textcoords="offset points", xytext=(8, 5), fontsize=9,
        )
    ax.axvline(x=alpha_theory, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectral Entropy alpha', fontsize=13)
    ax.set_ylabel('Lempel-Ziv Complexity (normalized)', fontsize=13)
    ax.set_title('Spectral Entropy alpha vs. Lempel-Ziv Complexity', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/alpha_vs_lz.png", dpi=200)
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/alpha_vs_lz.png")

    # --- Plot 6: Sensitivity analysis ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in key_constants:
        if name not in all_results:
            continue
        c = CONSTANTS[name]
        sens = all_results[name]['sensitivity']
        lbs = [s[0] for s in sens]
        als = [s[1] for s in sens]
        ax.plot(lbs, als, 'o-', label=c['display'], linewidth=1.5, markersize=5)
    ax.axhline(
        y=alpha_theory, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
        label=f'Theory ({alpha_theory:.3f})',
    )
    ax.set_xlabel('Lower fitting bound', fontsize=13)
    ax.set_ylabel('Fitted alpha', fontsize=13)
    ax.set_title('Sensitivity of alpha to Fitting Range', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sensitivity_analysis.png", dpi=200)
    plt.close()
    print(f"  Saved {OUTPUT_DIR}/sensitivity_analysis.png")

    # ---- Save raw data to text file ----
    print("\nSaving numerical results...")
    with open(f"{OUTPUT_DIR}/results_summary.txt", 'w') as f:
        f.write("SPECTRAL ENTROPY ANALYSIS - FULL RESULTS\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fast constants: {N_DIGITS:,} digits\n")
        f.write(f"Slow constants: {SLOW_DIGITS:,} digits\n")
        f.write(f"Theory alpha: {alpha_theory:.6f}\n")
        f.write(f"Fitting lower bound: {FIT_LOWER}\n\n")

        f.write(
            f"{'Constant':<18} {'Display':<10} {'alpha':>8} {'95CI':>8} {'R2':>8} "
            f"{'H_final':>8} {'LZ_norm':>10} {'N_digits':>10} {'Normal?':<25} {'Type'}\n"
        )
        f.write("-" * 140 + "\n")

        for name in sorted_names:
            r = all_results[name]
            c = CONSTANTS[name]
            f.write(
                f"{name:<18} {c['display']:<10} {r['alpha']:>8.4f} {r['ci95']:>8.4f} "
                f"{r['r2']:>8.5f} {r['H_final']:>8.4f} {r['lz_norm']:>10.5f} "
                f"{r['n_digits']:>10,} {c['normal']:<25} {c['type']}\n"
            )

        f.write("\n\nSENSITIVITY ANALYSIS\n")
        f.write("-" * 90 + "\n")
        for name in key_constants:
            if name not in all_results:
                continue
            f.write(f"\n{CONSTANTS[name]['display']}:\n")
            for lb, alpha_s, r2_s, ci_s in all_results[name]['sensitivity']:
                f.write(f"  N >= {lb:>6}: alpha = {alpha_s:.4f} +/- {ci_s:.4f}, R2 = {r2_s:.5f}\n")

    print(f"  Saved {OUTPUT_DIR}/results_summary.txt")

    print("\n" + "=" * 70)
    print("DONE! All results saved to:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
