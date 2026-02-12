#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Use mpmath for high-precision digit generation
from mpmath import mp, mpf, pi as mp_pi, e as mp_e, sqrt as mp_sqrt

# Configuration
NUM_DIGITS = 5000000  # Number of digits to analyze
BASES = [2, 4, 8, 10, 16]
CONSTANTS = ['pi', 'e', 'root2', 'champernowne', 'random', 'periodic']
OUTPUT_DIR = '/mnt/user-data/outputs/entropy_plots_new'

# Set precision for mpmath
mp.dps = NUM_DIGITS + 100  # Extra precision for safety

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def add_grids(ax, major_alpha=0.4, minor_alpha=0.15):
    """Add major and minor grids to axes."""
    ax.grid(True, which='major', linestyle='-', alpha=major_alpha, linewidth=0.8)
    ax.grid(True, which='minor', linestyle=':', alpha=minor_alpha, linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

def convert_to_base(number_str, from_base, to_base, num_digits):
    """Convert a number from one base to another."""
    if from_base == to_base:
        return number_str[:num_digits]
    
    # This is handled by getting digits directly in each base
    return number_str[:num_digits]

def get_pi_digits(base, num_digits):
    """Get digits of pi in specified base."""
    mp.dps = num_digits + 100
    pi_val = mp_pi
    
    # Remove integer part and get fractional digits
    frac = pi_val - int(pi_val)
    
    digits = []
    for _ in range(num_digits):
        frac *= base
        digit = int(frac)
        digits.append(digit)
        frac -= digit
    
    return digits

def get_e_digits(base, num_digits):
    """Get digits of e in specified base."""
    mp.dps = num_digits + 100
    e_val = mp_e
    
    frac = e_val - int(e_val)
    
    digits = []
    for _ in range(num_digits):
        frac *= base
        digit = int(frac)
        digits.append(digit)
        frac -= digit
    
    return digits

def get_root2_digits(base, num_digits):
    """Get digits of sqrt(2) in specified base."""
    mp.dps = num_digits + 100
    root2 = mp_sqrt(2)
    
    frac = root2 - int(root2)
    
    digits = []
    for _ in range(num_digits):
        frac *= base
        digit = int(frac)
        digits.append(digit)
        frac -= digit
    
    return digits

def get_champernowne_digits(base, num_digits):
    digits = []
    n = 1
    while len(digits) < num_digits:
        # Convert n to given base
        temp = n
        n_digits = []
        while temp > 0:
            n_digits.append(temp % base)
            temp //= base
        n_digits.reverse()
        digits.extend(n_digits)
        n += 1
    
    return digits[:num_digits]

def get_random_digits(base, num_digits, seed=42):
    """Get random digits (uniform distribution) in specified base."""
    np.random.seed(seed)
    return list(np.random.randint(0, base, num_digits))

def get_periodic_digits(base, num_digits):
    """
    Get periodic digits (repeating pattern) in specified base.
    Pattern: 0, 1, 2, ..., base-1, 0, 1, 2, ...
    """
    pattern = list(range(base))
    repeats = (num_digits // base) + 1
    full_pattern = pattern * repeats
    return full_pattern[:num_digits]

def get_digits(constant_name, base, num_digits):
    """Get digits for a given constant in specified base."""
    if constant_name == 'pi':
        return get_pi_digits(base, num_digits)
    elif constant_name == 'e':
        return get_e_digits(base, num_digits)
    elif constant_name == 'root2':
        return get_root2_digits(base, num_digits)
    elif constant_name == 'champernowne':
        return get_champernowne_digits(base, num_digits)
    elif constant_name == 'random':
        return get_random_digits(base, num_digits)
    elif constant_name == 'periodic':
        return get_periodic_digits(base, num_digits)
    else:
        raise ValueError(f"Unknown constant: {constant_name}")

def compute_digit_frequencies(digits, base):
    """Compute frequency of each digit."""
    counts = np.zeros(base)
    for d in digits:
        counts[d] += 1
    return counts / len(digits)

def compute_shannon_entropy(digits, base):
    """Compute Shannon entropy of digit sequence."""
    freqs = compute_digit_frequencies(digits, base)
    # Avoid log(0) by filtering zero frequencies
    freqs = freqs[freqs > 0]
    return -np.sum(freqs * np.log2(freqs))

def compute_shannon_entropy_cumulative(digits, base, step=100):
    """Compute cumulative Shannon entropy as digits increase."""
    entropies = []
    n_values = []
    
    for n in range(step, len(digits) + 1, step):
        H = compute_shannon_entropy(digits[:n], base)
        entropies.append(H)
        n_values.append(n)
    
    return np.array(n_values), np.array(entropies)

def compute_hydrodynamic_entropy(digits, base):
    # Block size for correlation analysis
    block_size = min(3, base)
    
    # Count block frequencies
    blocks = {}
    for i in range(len(digits) - block_size + 1):
        block = tuple(digits[i:i+block_size])
        blocks[block] = blocks.get(block, 0) + 1
    
    total_blocks = sum(blocks.values())
    freqs = np.array(list(blocks.values())) / total_blocks
    freqs = freqs[freqs > 0]
    
    # Hydrodynamic entropy as conditional entropy approximation
    block_entropy = -np.sum(freqs * np.log2(freqs))
    single_entropy = compute_shannon_entropy(digits, base)
    
    # Normalize by theoretical maximum
    max_block_entropy = block_size * np.log2(base)
    
    return block_entropy / max_block_entropy * single_entropy

def compute_hydrodynamic_entropy_cumulative(digits, base, step=100):
    entropies = []
    n_values = []
    
    for n in range(step, len(digits) + 1, step):
        if n >= 10:  # Need minimum digits for block entropy
            H = compute_hydrodynamic_entropy(digits[:n], base)
            entropies.append(H)
            n_values.append(n)
    
    return np.array(n_values), np.array(entropies)

def find_saturation_point(n_values, entropies, threshold=0.001, window=20):
    if len(entropies) < window + 5:
        return None, None
    
    # Smooth the data with specified window
    smoothed = np.convolve(entropies, np.ones(window)/window, mode='valid')
    
    # Compute derivative
    derivative = np.abs(np.diff(smoothed))
    
    # Find where derivative becomes consistently small (within tolerance)
    max_entropy = np.max(entropies)
    norm_threshold = threshold * max_entropy
    
    # Check for sustained low derivative over window
    for i in range(len(derivative) - window):
        if np.all(derivative[i:i+window] < norm_threshold):
            idx = i + window // 2
            if idx < len(n_values):
                return n_values[idx], entropies[idx]
    
    return None, None


def find_saturation_point_gradient(n_values, entropies, threshold=0.001, window=20):
    if len(entropies) < window + 10 or len(n_values) < window + 10:
        return None, None, None, None
    
    n_arr = np.array(n_values)
    e_arr = np.array(entropies)
    
    # Convert N to log10(N) for analysis
    log_n = np.log10(n_arr)
    
    # Compute first derivative (gradient): dH/d(log10(N))
    gradients = np.diff(e_arr) / np.diff(log_n)
    
    # Smooth the gradients
    if len(gradients) < window:
        return None, None, None, None
    
    smoothed_grad = np.convolve(gradients, np.ones(window)/window, mode='valid')
    
    # Compute second derivative (rate of change of gradient)
    second_derivative = np.abs(np.diff(smoothed_grad))
    
    # Find where second derivative becomes consistently small (gradient is constant)
    # Normalize threshold by the mean gradient magnitude
    mean_grad = np.mean(np.abs(smoothed_grad))
    if mean_grad > 0:
        norm_threshold = threshold * mean_grad
    else:
        norm_threshold = threshold
    
    # Check for sustained low second derivative over window
    for i in range(len(second_derivative) - window):
        if np.all(second_derivative[i:i+window] < norm_threshold):
            # Found saturation point
            idx = i + window // 2
            if idx < len(n_values):
                sat_n = n_values[idx]
                sat_e = entropies[idx]
                
                # Compute linear regression from saturation point to end
                # Using H vs log10(N)
                mask = n_arr >= sat_n
                if np.sum(mask) >= 2:
                    log_n_fit = np.log10(n_arr[mask])
                    e_fit = e_arr[mask]
                    # Linear regression: H = slope * log10(N) + intercept
                    slope, intercept = np.polyfit(log_n_fit, e_fit, 1)
                    return sat_n, sat_e, slope, intercept
                
                return sat_n, sat_e, None, None
    
    return None, None, None, None


def compute_linear_fit_from_saturation(n_values, entropies, sat_n):
    n_arr = np.array(n_values)
    e_arr = np.array(entropies)
    
    mask = n_arr >= sat_n
    if np.sum(mask) < 2:
        return None, None, None
    
    n_fit = n_arr[mask]
    e_fit = e_arr[mask]
    
    # Linear regression
    slope, intercept = np.polyfit(n_fit, e_fit, 1)
    
    # R-squared
    predicted = slope * n_fit + intercept
    ss_res = np.sum((e_fit - predicted) ** 2)
    ss_tot = np.sum((e_fit - np.mean(e_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared

def compute_digit_resolve_spectra(digits, base):
    # Normalize digits to [-1, 1] range
    normalized = (np.array(digits) - (base - 1) / 2) / ((base - 1) / 2)
    
    # Compute FFT
    N = len(normalized)
    fft_vals = fft(normalized)
    freqs = fftfreq(N)
    
    # Power spectrum (positive frequencies only)
    positive_freq_mask = freqs > 0
    power_spectrum = np.abs(fft_vals[positive_freq_mask]) ** 2
    frequencies = freqs[positive_freq_mask]
    
    return frequencies, power_spectrum

def plot_digit_frequency(digits, base, constant_name, output_dir):
    """Plot digit frequency distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    freqs = compute_digit_frequencies(digits, base)
    expected = 1.0 / base
    
    x = np.arange(base)
    bars = ax.bar(x, freqs * 100, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1)
    
    # Expected frequency line
    ax.axhline(y=expected * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Expected ({expected*100:.2f}%)')
    
    ax.set_xlabel(f'Digit (Base {base})')
    ax.set_ylabel('Frequency (%)')
    ax.set_title(f'Digit Frequency Distribution: {constant_name.upper()} (Base {base})\n'
                 f'N = {len(digits):,} digits')
    
    # Set x-ticks
    if base <= 16:
        labels = [format(i, 'X') if base == 16 else str(i) for i in range(base)]
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    
    add_grids(ax)
    ax.legend(loc='upper right')
    
    # Add percentage labels on bars
    for bar, freq in zip(bars, freqs):
        height = bar.get_height()
        ax.annotate(f'{freq*100:.2f}%',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'digit_frequency_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_digit_frequency_vs_n(digits, base, constant_name, output_dir):
    """Plot digit frequency percentage vs number of digits (cumulative)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    step = max(1, len(digits) // 200)
    n_values = list(range(step, len(digits) + 1, step))
    
    # Track frequencies for each digit over increasing N
    freq_history = {d: [] for d in range(base)}
    
    for n in n_values:
        freqs = compute_digit_frequencies(digits[:n], base)
        for d in range(base):
            freq_history[d].append(freqs[d] * 100)
    
    # Plot each digit's frequency
    colors = plt.cm.tab10(np.linspace(0, 1, base))
    for d in range(base):
        label = format(d, 'X') if base == 16 else str(d)
        ax.plot(n_values, freq_history[d], color=colors[d], 
               label=f'Digit {label}', alpha=0.8, linewidth=1.2)
    
    # Expected frequency line
    expected = 100.0 / base
    ax.axhline(y=expected, color='black', linestyle='--', linewidth=2, 
               label=f'Expected ({expected:.2f}%)')
    
    ax.set_xlabel('Number of Digits (N)')
    ax.set_ylabel('Frequency (%)')
    ax.set_title(f'Digit Frequency vs N: {constant_name.upper()} (Base {base})')
    
    add_grids(ax)
    ax.legend(loc='upper right', ncol=min(4, (base + 2) // 2), fontsize=8)
    ax.set_xlim(0, len(digits))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'digit_frequency_vs_n_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_digit_resolve_spectra(digits, base, constant_name, output_dir):
    """Plot digit resolve spectra (power spectrum)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    freqs, power = compute_digit_resolve_spectra(digits, base)
    
    # Plot power spectrum (log scale for better visualization)
    ax.semilogy(freqs[:len(freqs)//4], power[:len(power)//4], 
                color='purple', alpha=0.7, linewidth=0.5)
    
    # Find and mark peaks
    peak_indices, _ = find_peaks(np.log10(power[:len(power)//4] + 1e-10), height=-5, distance=50)
    if len(peak_indices) > 0:
        top_peaks = peak_indices[:min(5, len(peak_indices))]
        ax.scatter(freqs[top_peaks], power[top_peaks], color='red', s=50, 
                  zorder=5, label='Major Peaks')
    
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power Spectral Density (log scale)')
    ax.set_title(f'Digit Resolve Spectra: {constant_name.upper()} (Base {base})\n'
                 f'N = {len(digits):,} digits')
    
    add_grids(ax)
    if len(peak_indices) > 0:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'digit_resolve_spectra_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_shannon_entropy(digits, base, constant_name, output_dir):
    """Plot Shannon entropy vs number of digits with saturation point and linear fit."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    step = max(1, len(digits) // 200)
    n_values, entropies = compute_shannon_entropy_cumulative(digits, base, step)
    
    # Maximum theoretical entropy
    max_entropy = np.log2(base)
    
    # Plot entropy curve
    ax.plot(n_values, entropies, color='blue', linewidth=1.5, label='Shannon Entropy')
    
    # Plot theoretical maximum
    ax.axhline(y=max_entropy, color='red', linestyle='--', linewidth=1.5,
               label=f'Theoretical Max ({max_entropy:.4f} bits)')
    
    # Find and mark saturation point (tolerance=0.001, window=20)
    sat_n, sat_entropy = find_saturation_point(n_values, entropies, threshold=0.001, window=20)
    slope, intercept, r_sq = None, None, None
    
    if sat_n is not None:
        ax.scatter([sat_n], [sat_entropy], color='green', s=100, zorder=5, 
                  marker='o', edgecolors='darkgreen', linewidths=2,
                  label=f'Saturation: N={sat_n}')
        # Add N value label close to the point
        ax.annotate(f'N={sat_n}', xy=(sat_n, sat_entropy), 
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=10, color='darkgreen', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Linear fit from saturation point to end
        slope, intercept, r_sq = compute_linear_fit_from_saturation(n_values, entropies, sat_n)
        if slope is not None:
            n_arr = np.array(n_values)
            mask = n_arr >= sat_n
            n_fit = n_arr[mask]
            fit_line = slope * n_fit + intercept
            ax.plot(n_fit, fit_line, color='magenta', linewidth=2, linestyle='-.',
                   label=f'Linear Fit: slope={slope:.2e}, R²={r_sq:.4f}')
    
    ax.set_xlabel('Number of Digits (N)')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title(f'Shannon Entropy vs N: {constant_name.upper()} (Base {base})')
    
    add_grids(ax)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, len(digits))
    ax.set_ylim(0, max_entropy * 1.1)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'shannon_entropy_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath, {'sat_n': sat_n, 'sat_entropy': sat_entropy, 'slope': slope, 'intercept': intercept, 'r_squared': r_sq}

def plot_hydrodynamic_entropy(digits, base, constant_name, output_dir):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    step = max(1, len(digits) // 200)
    n_values, entropies = compute_hydrodynamic_entropy_cumulative(digits, base, step)
    
    if len(entropies) == 0:
        plt.close()
        return None, {}
    
    # Convert to log10(N) for x-axis
    log_n = np.log10(np.array(n_values))
    
    # Plot entropy curve: H vs log10(N)
    ax.plot(log_n, entropies, color='darkorange', linewidth=1.5, label='Hydrodynamic Entropy')
    
    # Reference: Shannon entropy for comparison
    _, shannon_entropies = compute_shannon_entropy_cumulative(digits, base, step)
    ax.plot(log_n[:len(shannon_entropies)], shannon_entropies, 
           color='blue', linewidth=1, alpha=0.5, linestyle=':', label='Shannon Entropy (ref)')
    
    # Check if this is a deterministic constant (Champernowne or Periodic)
    # For these, skip saturation detection and fit line from the start
    deterministic_constants = ['champernowne', 'periodic']
    
    sat_n, sat_entropy, slope, intercept = None, None, None, None
    
    if constant_name.lower() in deterministic_constants:
        # Fit line from the start (no saturation point)
        slope, intercept = np.polyfit(log_n, entropies, 1)
        fit_line = slope * log_n + intercept
        ax.plot(log_n, fit_line, color='magenta', linewidth=2, linestyle='-.',
               label=f'Linear Fit (full): slope={slope:.4f}, intercept={intercept:.4f}')
        # Note: sat_n remains None for deterministic constants
    else:
        # Find saturation point using gradient method (where gradient becomes constant)
        sat_n, sat_entropy, slope, intercept = find_saturation_point_gradient(n_values, entropies, threshold=0.001, window=20)
        
        if sat_n is not None:
            sat_log_n = np.log10(sat_n)
            ax.scatter([sat_log_n], [sat_entropy], color='purple', s=100, zorder=5,
                      marker='o', edgecolors='darkviolet', linewidths=2,
                      label=f'Saturation: N={sat_n}')
            # Add N value label close to the point
            ax.annotate(f'N={sat_n}', xy=(sat_log_n, sat_entropy),
                       xytext=(10, -15), textcoords='offset points',
                       fontsize=10, color='darkviolet', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='plum', alpha=0.7))
            
            # Plot linear fit from saturation to end: H = slope * log10(N) + intercept
            if slope is not None:
                n_arr = np.array(n_values)
                mask = n_arr >= sat_n
                log_n_fit = np.log10(n_arr[mask])
                fit_line = slope * log_n_fit + intercept
                ax.plot(log_n_fit, fit_line, color='magenta', linewidth=2, linestyle='-.',
                       label=f'Linear Fit: slope={slope:.4f}, intercept={intercept:.4f}')
        else:
            # No saturation found, fit from start as fallback
            slope, intercept = np.polyfit(log_n, entropies, 1)
            fit_line = slope * log_n + intercept
            ax.plot(log_n, fit_line, color='magenta', linewidth=2, linestyle='-.',
                   label=f'Linear Fit (full): slope={slope:.4f}, intercept={intercept:.4f}')
    
    ax.set_xlabel('log₁₀(N)')
    ax.set_ylabel('Hydrodynamic Entropy')
    ax.set_title(f'Hydrodynamic Entropy vs log₁₀(N): {constant_name.upper()} (Base {base})')
    
    add_grids(ax)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'hydrodynamic_entropy_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath, {'sat_n': sat_n, 'sat_entropy': sat_entropy, 'slope': slope, 'intercept': intercept}

def plot_entropy_comparison(digits, base, constant_name, output_dir):
    """Plot comparison of Shannon and Hydrodynamic entropy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    step = max(1, len(digits) // 200)
    n_s, shannon = compute_shannon_entropy_cumulative(digits, base, step)
    n_h, hydro = compute_hydrodynamic_entropy_cumulative(digits, base, step)
    
    # Shannon entropy subplot (H vs N)
    ax1.plot(n_s, shannon, color='blue', linewidth=1.5, label='Shannon Entropy')
    max_s = np.log2(base)
    ax1.axhline(y=max_s, color='red', linestyle='--', alpha=0.7, label=f'Max ({max_s:.3f})')
    
    sat_n_s, sat_e_s = find_saturation_point(n_s, shannon, threshold=0.001, window=20)
    if sat_n_s:
        ax1.scatter([sat_n_s], [sat_e_s], color='green', s=80, zorder=5,
                   label=f'Saturation: N={sat_n_s}')
        ax1.annotate(f'N={sat_n_s}', xy=(sat_n_s, sat_e_s),
                    xytext=(5, -10), textcoords='offset points',
                    fontsize=9, color='darkgreen', fontweight='bold')
        # Linear fit
        slope_s, intercept_s, _ = compute_linear_fit_from_saturation(n_s, shannon, sat_n_s)
        if slope_s is not None:
            n_arr = np.array(n_s)
            mask = n_arr >= sat_n_s
            fit_line = slope_s * n_arr[mask] + intercept_s
            ax1.plot(n_arr[mask], fit_line, 'magenta', linewidth=1.5, linestyle='-.', 
                    label=f'Fit: slope={slope_s:.2e}')
    
    ax1.set_xlabel('Number of Digits (N)')
    ax1.set_ylabel('Shannon Entropy (bits)')
    ax1.set_title('Shannon Entropy vs N')
    ax1.legend(loc='lower right', fontsize=8)
    add_grids(ax1)
    
    # Hydrodynamic entropy subplot (H vs log10(N))
    deterministic_constants = ['champernowne', 'periodic']
    
    if len(hydro) > 0:
        log_n_h = np.log10(np.array(n_h))
        ax2.plot(log_n_h, hydro, color='darkorange', linewidth=1.5, label='Hydrodynamic Entropy')
        
        if constant_name.lower() in deterministic_constants:
            # Fit line from start for deterministic constants
            slope_h, intercept_h = np.polyfit(log_n_h, hydro, 1)
            fit_line = slope_h * log_n_h + intercept_h
            ax2.plot(log_n_h, fit_line, 'magenta', linewidth=1.5, linestyle='-.', 
                    label=f'Fit (full): slope={slope_h:.4f}')
        else:
            sat_n_h, sat_e_h, slope_h, intercept_h = find_saturation_point_gradient(n_h, hydro, threshold=0.001, window=20)
            if sat_n_h:
                sat_log_n = np.log10(sat_n_h)
                ax2.scatter([sat_log_n], [sat_e_h], color='purple', s=80, zorder=5,
                           label=f'Saturation: N={sat_n_h}')
                ax2.annotate(f'N={sat_n_h}', xy=(sat_log_n, sat_e_h),
                            xytext=(5, -10), textcoords='offset points',
                            fontsize=9, color='darkviolet', fontweight='bold')
                # Linear fit
                if slope_h is not None:
                    n_arr = np.array(n_h)
                    mask = n_arr >= sat_n_h
                    log_n_fit = np.log10(n_arr[mask])
                    fit_line = slope_h * log_n_fit + intercept_h
                    ax2.plot(log_n_fit, fit_line, 'magenta', linewidth=1.5, linestyle='-.', 
                            label=f'Fit: slope={slope_h:.4f}')
        ax2.legend(loc='lower right', fontsize=8)
    
    ax2.set_xlabel('log₁₀(N)')
    ax2.set_ylabel('Hydrodynamic Entropy')
    ax2.set_title('Hydrodynamic Entropy vs log₁₀(N)')
    add_grids(ax2)
    
    fig.suptitle(f'Entropy Comparison: {constant_name.upper()} (Base {base})', fontsize=14, y=1.02)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'entropy_comparison_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_chi_square_analysis(digits, base, constant_name, output_dir):
    """Plot chi-square statistic vs N for uniformity testing."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    step = max(1, len(digits) // 200)
    n_values = list(range(step, len(digits) + 1, step))
    chi_squares = []
    
    expected = 1.0 / base
    
    for n in n_values:
        freqs = compute_digit_frequencies(digits[:n], base)
        chi_sq = n * np.sum((freqs - expected) ** 2 / expected)
        chi_squares.append(chi_sq)
    
    ax.plot(n_values, chi_squares, color='teal', linewidth=1.5)
    
    # Critical value for chi-square (df = base - 1, alpha = 0.05)
    from scipy.stats import chi2
    critical = chi2.ppf(0.95, df=base - 1)
    ax.axhline(y=critical, color='red', linestyle='--', linewidth=1.5,
               label=f'Critical Value (α=0.05): {critical:.2f}')
    
    ax.set_xlabel('Number of Digits (N)')
    ax.set_ylabel('Chi-Square Statistic')
    ax.set_title(f'Chi-Square Uniformity Test: {constant_name.upper()} (Base {base})')
    
    add_grids(ax)
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(digits))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'chi_square_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_autocorrelation(digits, base, constant_name, output_dir):
    """Plot autocorrelation of digit sequence."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize digits
    normalized = np.array(digits) - np.mean(digits)
    
    # Compute autocorrelation
    max_lag = min(500, len(digits) // 10)
    autocorr = np.correlate(normalized, normalized, mode='full')
    autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + max_lag]
    autocorr = autocorr / autocorr[0]  # Normalize
    
    lags = np.arange(max_lag)
    ax.plot(lags, autocorr, color='navy', linewidth=1)
    
    # Confidence bounds (approximate)
    confidence = 1.96 / np.sqrt(len(digits))
    ax.axhline(y=confidence, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=-confidence, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Autocorrelation: {constant_name.upper()} (Base {base})')
    
    add_grids(ax)
    ax.set_xlim(0, max_lag)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, f'autocorrelation_base{base}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    return filepath

def plot_summary_dashboard(all_results, constant_name, output_dir):
    """Create a summary dashboard with key metrics across all bases."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    bases = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bases)))
    
    # Plot 1: Final Shannon entropy vs Base
    ax = axes[0, 0]
    final_shannon = [all_results[b]['final_shannon'] for b in bases]
    max_shannon = [np.log2(b) for b in bases]
    
    x = np.arange(len(bases))
    width = 0.35
    ax.bar(x - width/2, final_shannon, width, label='Observed', color='steelblue')
    ax.bar(x + width/2, max_shannon, width, label='Maximum', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bases])
    ax.set_xlabel('Base')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_title('Final Shannon Entropy by Base')
    ax.legend()
    add_grids(ax)
    
    # Plot 2: Saturation N values
    ax = axes[0, 1]
    sat_n_shannon = [all_results[b].get('sat_n_shannon', 0) or 0 for b in bases]
    sat_n_hydro = [all_results[b].get('sat_n_hydro', 0) or 0 for b in bases]
    
    ax.bar(x - width/2, sat_n_shannon, width, label='Shannon', color='blue', alpha=0.7)
    ax.bar(x + width/2, sat_n_hydro, width, label='Hydrodynamic', color='orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bases])
    ax.set_xlabel('Base')
    ax.set_ylabel('Saturation N')
    ax.set_title('Saturation Point (N) by Base')
    ax.legend()
    add_grids(ax)
    
    # Plot 3: Chi-square final values
    ax = axes[0, 2]
    chi_sq = [all_results[b]['final_chi_square'] for b in bases]
    critical_vals = [all_results[b]['chi_critical'] for b in bases]
    
    ax.bar(x, chi_sq, color='teal', alpha=0.8, label='Observed χ²')
    ax.scatter(x, critical_vals, color='red', s=100, marker='_', linewidths=3, 
               label='Critical (α=0.05)', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bases])
    ax.set_xlabel('Base')
    ax.set_ylabel('Chi-Square Statistic')
    ax.set_title('Uniformity Test (χ²) by Base')
    ax.legend()
    add_grids(ax)
    
    # Plot 4: Entropy efficiency (observed/max)
    ax = axes[1, 0]
    efficiency = [all_results[b]['final_shannon'] / np.log2(b) * 100 for b in bases]
    colors_eff = ['green' if e > 99 else 'orange' if e > 95 else 'red' for e in efficiency]
    ax.bar(x, efficiency, color=colors_eff, alpha=0.8)
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bases])
    ax.set_xlabel('Base')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Shannon Entropy Efficiency')
    ax.set_ylim(90, 101)
    add_grids(ax)
    
    # Plot 5: Digit frequency deviation (max deviation from expected)
    ax = axes[1, 1]
    max_devs = [all_results[b]['max_freq_deviation'] * 100 for b in bases]
    ax.bar(x, max_devs, color='purple', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bases])
    ax.set_xlabel('Base')
    ax.set_ylabel('Max Deviation (%)')
    ax.set_title('Maximum Frequency Deviation')
    add_grids(ax)
    
    # Plot 6: Text summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"Summary: {constant_name.upper()}\n"
    summary_text += f"{'='*40}\n"
    summary_text += f"Total Digits Analyzed: {NUM_DIGITS:,}\n\n"
    
    for b in bases:
        res = all_results[b]
        summary_text += f"Base {b}:\n"
        summary_text += f"  Shannon: {res['final_shannon']:.4f} bits\n"
        summary_text += f"  Sat. N (S/H): {res.get('sat_n_shannon', 'N/A')}/{res.get('sat_n_hydro', 'N/A')}\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(f'Analysis Dashboard: {constant_name.upper()}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'summary_dashboard.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath

def analyze_constant_base(args):
    """
    Analyze a single constant-base combination.
    This function is designed for parallel processing.
    """
    constant_name, base = args
    
    print(f"  Processing {constant_name} in base {base}...")
    
    try:
        # Get digits
        digits = get_digits(constant_name, base, NUM_DIGITS)
        
        # Compute final metrics
        final_shannon = compute_shannon_entropy(digits, base)
        final_hydro = compute_hydrodynamic_entropy(digits, base)
        
        # Saturation points
        step = max(1, NUM_DIGITS // 200)
        n_s, e_s = compute_shannon_entropy_cumulative(digits, base, step)
        n_h, e_h = compute_hydrodynamic_entropy_cumulative(digits, base, step)
        
        sat_n_shannon, _ = find_saturation_point(n_s, e_s, threshold=0.001, window=20)
        sat_n_hydro, _, _, _ = find_saturation_point_gradient(n_h, e_h, threshold=0.001, window=20) if len(e_h) > 0 else (None, None, None, None)
        
        # Chi-square
        from scipy.stats import chi2
        freqs = compute_digit_frequencies(digits, base)
        expected = 1.0 / base
        final_chi_sq = NUM_DIGITS * np.sum((freqs - expected) ** 2 / expected)
        chi_critical = chi2.ppf(0.95, df=base - 1)
        
        # Max frequency deviation
        max_freq_dev = np.max(np.abs(freqs - expected))
        
        return {
            'constant': constant_name,
            'base': base,
            'digits': digits,
            'final_shannon': final_shannon,
            'final_hydro': final_hydro,
            'sat_n_shannon': sat_n_shannon,
            'sat_n_hydro': sat_n_hydro,
            'final_chi_square': final_chi_sq,
            'chi_critical': chi_critical,
            'max_freq_deviation': max_freq_dev
        }
    except Exception as e:
        print(f"  Error processing {constant_name} base {base}: {e}")
        return None

def generate_all_plots(result, output_dir):
    """Generate all plots for a single constant-base combination."""
    constant_name = result['constant']
    base = result['base']
    digits = result['digits']
    
    plots = []
    fit_data = {}
    
    # Generate all plot types
    plots.append(plot_digit_frequency(digits, base, constant_name, output_dir))
    plots.append(plot_digit_frequency_vs_n(digits, base, constant_name, output_dir))
    plots.append(plot_digit_resolve_spectra(digits, base, constant_name, output_dir))
    
    # Shannon entropy - returns (filepath, fit_info)
    shannon_result = plot_shannon_entropy(digits, base, constant_name, output_dir)
    plots.append(shannon_result[0])
    fit_data['shannon'] = shannon_result[1]
    
    # Hydrodynamic entropy - returns (filepath, fit_info)
    hydro_result = plot_hydrodynamic_entropy(digits, base, constant_name, output_dir)
    if hydro_result[0]:
        plots.append(hydro_result[0])
    fit_data['hydrodynamic'] = hydro_result[1]
    
    plots.append(plot_entropy_comparison(digits, base, constant_name, output_dir))
    plots.append(plot_chi_square_analysis(digits, base, constant_name, output_dir))
    plots.append(plot_autocorrelation(digits, base, constant_name, output_dir))
    
    return [p for p in plots if p is not None], fit_data

def process_constant(constant_name):
    """Process a single constant across all bases."""
    print(f"\nProcessing constant: {constant_name}")
    
    # Create output directory for this constant
    constant_dir = os.path.join(OUTPUT_DIR, constant_name)
    os.makedirs(constant_dir, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = [(constant_name, base) for base in BASES]
    
    # Process all bases
    results = {}
    for args in args_list:
        result = analyze_constant_base(args)
        if result:
            results[result['base']] = result
    
    # Generate plots for each base and collect fit data
    all_plots = []
    all_fit_data = {}
    
    for base in BASES:
        if base in results:
            plots, fit_data = generate_all_plots(results[base], constant_dir)
            all_plots.extend(plots)
            all_fit_data[base] = fit_data
    
    # Generate summary dashboard
    if results:
        summary_plot = plot_summary_dashboard(results, constant_name, constant_dir)
        all_plots.append(summary_plot)
    
    # Export fit data to CSV for this constant
    export_fit_data_csv(constant_name, all_fit_data, constant_dir)
    
    return constant_name, results, all_plots, all_fit_data


def export_fit_data_csv(constant_name, fit_data, output_dir):
    """Export linear fit data (slopes, intercepts) to CSV file."""
    csv_path = os.path.join(output_dir, f'{constant_name}_linear_fit_data.csv')
    
    with open(csv_path, 'w') as f:
        f.write("Constant,Base,Entropy_Type,Saturation_N,Saturation_Entropy,Slope,Intercept,R_Squared\n")
        
        for base in sorted(fit_data.keys()):
            base_data = fit_data[base]
            
            # Shannon entropy data
            shannon = base_data.get('shannon', {})
            sat_n_s = shannon.get('sat_n', '')
            sat_e_s = shannon.get('sat_entropy', '')
            slope_s = shannon.get('slope', '')
            intercept_s = shannon.get('intercept', '')
            r_sq_s = shannon.get('r_squared', '')
            
            # Format values
            sat_n_s = sat_n_s if sat_n_s is not None else ''
            sat_e_s = f"{sat_e_s:.6f}" if sat_e_s is not None else ''
            slope_s = f"{slope_s:.6e}" if slope_s is not None else ''
            intercept_s = f"{intercept_s:.6f}" if intercept_s is not None else ''
            r_sq_s = f"{r_sq_s:.6f}" if r_sq_s is not None else ''
            
            f.write(f"{constant_name},{base},Shannon,{sat_n_s},{sat_e_s},{slope_s},{intercept_s},{r_sq_s}\n")
            
            # Hydrodynamic entropy data
            hydro = base_data.get('hydrodynamic', {})
            sat_n_h = hydro.get('sat_n', '')
            sat_e_h = hydro.get('sat_entropy', '')
            slope_h = hydro.get('slope', '')
            intercept_h = hydro.get('intercept', '')
            
            # Format values
            sat_n_h = sat_n_h if sat_n_h is not None else ''
            sat_e_h = f"{sat_e_h:.6f}" if sat_e_h is not None else ''
            slope_h = f"{slope_h:.6f}" if slope_h is not None else ''
            intercept_h = f"{intercept_h:.6f}" if intercept_h is not None else ''
            
            f.write(f"{constant_name},{base},Hydrodynamic,{sat_n_h},{sat_e_h},{slope_h},{intercept_h},\n")
    
    print(f"  Saved fit data to {csv_path}")

def main():
    """Main execution function with parallelization."""
    print("="*60)
    print("DIGIT ENTROPY ANALYSIS")
    print("="*60)
    print(f"Constants: {', '.join(CONSTANTS)}")
    print(f"Bases: {', '.join(map(str, BASES))}")
    print(f"Digits per analysis: {NUM_DIGITS:,}")
    print(f"Available CPUs: {cpu_count()}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Setup
    setup_plot_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each constant (parallelization is within each constant across bases)
    all_results = {}
    all_fit_data = {}
    
    # Use multiprocessing pool for parallel processing across constants
    n_workers = min(cpu_count(), len(CONSTANTS))
    print(f"\nUsing {n_workers} parallel workers...")
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_constant, CONSTANTS)
    
    # Collect results
    for constant_name, const_results, plots, fit_data in results:
        all_results[constant_name] = const_results
        all_fit_data[constant_name] = fit_data
        print(f"  {constant_name}: Generated {len(plots)} plots")
    
    # Create master comparison plots
    print("\nGenerating master comparison plots...")
    create_master_comparisons(all_results)
    
    # Create master CSV with all fit data
    create_master_fit_csv(all_fit_data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)
    
    # Print summary
    print("\nSUMMARY OF RESULTS:")
    print("-"*60)
    for constant in CONSTANTS:
        if constant in all_results:
            print(f"\n{constant.upper()}:")
            for base in BASES:
                if base in all_results[constant]:
                    r = all_results[constant][base]
                    print(f"  Base {base}: Shannon={r['final_shannon']:.4f}, "
                          f"SatN_S={r['sat_n_shannon']}, SatN_H={r['sat_n_hydro']}")


def create_master_fit_csv(all_fit_data):
    """Create a master CSV file with all linear fit data for all constants."""
    csv_path = os.path.join(OUTPUT_DIR, 'master_linear_fit_data.csv')
    
    with open(csv_path, 'w') as f:
        f.write("Constant,Base,Entropy_Type,Saturation_N,Saturation_Entropy,Slope,Intercept,R_Squared,Notes\n")
        
        for constant in CONSTANTS:
            if constant not in all_fit_data:
                continue
            
            for base in BASES:
                if base not in all_fit_data[constant]:
                    continue
                
                base_data = all_fit_data[constant][base]
                
                # Shannon entropy data
                shannon = base_data.get('shannon', {})
                sat_n_s = shannon.get('sat_n')
                sat_e_s = shannon.get('sat_entropy')
                slope_s = shannon.get('slope')
                intercept_s = shannon.get('intercept')
                r_sq_s = shannon.get('r_squared')
                
                row_s = [
                    constant,
                    str(base),
                    'Shannon',
                    str(sat_n_s) if sat_n_s is not None else '',
                    f"{sat_e_s:.6f}" if sat_e_s is not None else '',
                    f"{slope_s:.6e}" if slope_s is not None else '',
                    f"{intercept_s:.6f}" if intercept_s is not None else '',
                    f"{r_sq_s:.6f}" if r_sq_s is not None else '',
                    'H vs N (plateaus)'
                ]
                f.write(','.join(row_s) + '\n')
                
                # Hydrodynamic entropy data
                hydro = base_data.get('hydrodynamic', {})
                sat_n_h = hydro.get('sat_n')
                sat_e_h = hydro.get('sat_entropy')
                slope_h = hydro.get('slope')
                intercept_h = hydro.get('intercept')
                
                row_h = [
                    constant,
                    str(base),
                    'Hydrodynamic',
                    str(sat_n_h) if sat_n_h is not None else '',
                    f"{sat_e_h:.6f}" if sat_e_h is not None else '',
                    f"{slope_h:.6f}" if slope_h is not None else '',
                    f"{intercept_h:.6f}" if intercept_h is not None else '',
                    '',
                    'H vs log10(N) (linear after saturation)'
                ]
                f.write(','.join(row_h) + '\n')
    
    print(f"  Master fit data saved to {csv_path}")

def create_master_comparisons(all_results):
    """Create comparison plots across all constants."""
    master_dir = os.path.join(OUTPUT_DIR, '_master_comparisons')
    os.makedirs(master_dir, exist_ok=True)
    
    # Plot 1: Shannon entropy comparison across all constants for each base
    for base in BASES:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        constants = []
        entropies = []
        
        for const in CONSTANTS:
            if const in all_results and base in all_results[const]:
                constants.append(const.upper())
                entropies.append(all_results[const][base]['final_shannon'])
        
        max_entropy = np.log2(base)
        
        x = np.arange(len(constants))
        bars = ax.bar(x, entropies, color='steelblue', alpha=0.8, edgecolor='navy')
        ax.axhline(y=max_entropy, color='red', linestyle='--', linewidth=2,
                   label=f'Max Entropy ({max_entropy:.3f} bits)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(constants, rotation=45, ha='right')
        ax.set_xlabel('Constant')
        ax.set_ylabel('Shannon Entropy (bits)')
        ax.set_title(f'Shannon Entropy Comparison - Base {base}')
        ax.legend()
        add_grids(ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(master_dir, f'comparison_shannon_base{base}.png'), dpi=150)
        plt.close()
    
    # Plot 2: Saturation N comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Shannon saturation
    ax = axes[0]
    for const in CONSTANTS:
        if const in all_results:
            bases_used = []
            sat_n_vals = []
            for base in BASES:
                if base in all_results[const]:
                    sat_n = all_results[const][base].get('sat_n_shannon')
                    if sat_n:
                        bases_used.append(base)
                        sat_n_vals.append(sat_n)
            if sat_n_vals:
                ax.plot(bases_used, sat_n_vals, 'o-', label=const.upper(), markersize=8)
    
    ax.set_xlabel('Base')
    ax.set_ylabel('Saturation N')
    ax.set_title('Shannon Entropy Saturation Points')
    ax.legend()
    add_grids(ax)
    
    # Hydrodynamic saturation
    ax = axes[1]
    for const in CONSTANTS:
        if const in all_results:
            bases_used = []
            sat_n_vals = []
            for base in BASES:
                if base in all_results[const]:
                    sat_n = all_results[const][base].get('sat_n_hydro')
                    if sat_n:
                        bases_used.append(base)
                        sat_n_vals.append(sat_n)
            if sat_n_vals:
                ax.plot(bases_used, sat_n_vals, 'o-', label=const.upper(), markersize=8)
    
    ax.set_xlabel('Base')
    ax.set_ylabel('Saturation N')
    ax.set_title('Hydrodynamic Entropy Saturation Points')
    ax.legend()
    add_grids(ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, 'saturation_comparison.png'), dpi=150)
    plt.close()
    
    # Plot 3: Efficiency heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    efficiency_matrix = np.zeros((len(CONSTANTS), len(BASES)))
    
    for i, const in enumerate(CONSTANTS):
        for j, base in enumerate(BASES):
            if const in all_results and base in all_results[const]:
                observed = all_results[const][base]['final_shannon']
                theoretical_max = np.log2(base)
                efficiency_matrix[i, j] = (observed / theoretical_max) * 100
    
    im = ax.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=90, vmax=100)
    
    ax.set_xticks(np.arange(len(BASES)))
    ax.set_yticks(np.arange(len(CONSTANTS)))
    ax.set_xticklabels([str(b) for b in BASES])
    ax.set_yticklabels([c.upper() for c in CONSTANTS])
    
    ax.set_xlabel('Base')
    ax.set_ylabel('Constant')
    ax.set_title('Shannon Entropy Efficiency (%) - Heatmap')
    
    # Add values as text
    for i in range(len(CONSTANTS)):
        for j in range(len(BASES)):
            val = efficiency_matrix[i, j]
            color = 'white' if val < 95 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   color=color, fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Efficiency (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(master_dir, 'efficiency_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"  Master comparisons saved to {master_dir}")

if __name__ == "__main__":
    main()
