import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import gc
import os
import secrets
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURATION ---
MASTER_FOLDER = 'entropy_plts_new'
N_DIGITS = 1_000_000  # Analysis depth
BASES = [2, 4, 8, 10, 16]
CONSTANTS = ['pi', 'e', 'sqrt2', 'champernowne', 'random', 'periodic']
TOLERANCE = 0.001
CONFIRMATION_WINDOW = 20

plt.style.use('seaborn-v0_8-darkgrid')

class ComprehensiveAnalyzer:
    def __init__(self, name, base, n_digits):
        self.name = name
        self.base = base
        self.n_digits = n_digits
        self.output_dir = os.path.join(MASTER_FOLDER, name, f'base_{base}')
        os.makedirs(self.output_dir, exist_ok=True)
        self.digits = None

    def generate_sequence(self):
        """Fixed digit generation using integer scaling for arbitrary bases."""
        if self.name in ['pi', 'e', 'sqrt2']:
            # For base b, we need floor(constant * b^n) to get n digits
            mp.dps = self.n_digits + 100
            if self.name == 'pi': val = mp.pi
            elif self.name == 'e': val = mp.e
            else: val = mp.sqrt(2)
            
            # Shift decimal point by n_digits in the target base
            # floor(val * base^n_digits) gives us the sequence as a massive integer
            multiplier = mp.mpf(self.base)**self.n_digits
            big_int = int(val * multiplier)
            
            # Convert integer to string in target base
            s = np.base_repr(big_int, base=self.base).lower()
            # Ensure we take the tail if the integer conversion trimmed leading zeros
            self.digits = np.array([int(c, self.base) for c in s[-self.n_digits:]], dtype=np.int8)

        elif self.name == 'champernowne':
            s = ""
            i = 1
            while len(s) < self.n_digits:
                s += np.base_repr(i, base=self.base)
                i += 1
            self.digits = np.array([int(c, self.base) for c in s[:self.n_digits]], dtype=np.int8)

        elif self.name == 'random':
            self.digits = np.array([secrets.randbelow(self.base) for _ in range(self.n_digits)], dtype=np.int8)

        elif self.name == 'periodic':
            pattern = np.arange(self.base, dtype=np.int8)
            self.digits = np.tile(pattern, (self.n_digits // self.base) + 1)[:self.n_digits]

        return self.digits

    def run_analysis(self):
        self.generate_sequence()
        h_max = np.log2(self.base)
        
        # Logarithmic sampling for performance
        N_values = np.logspace(2, np.log10(self.n_digits), 150, dtype=int)
        N_values = np.unique(N_values)

        h_shannon = []
        h_hydro = []
        freq_data = {d: [] for d in range(self.base)}
        
        # Hydrodynamic window (local entropy)
        hydro_win = 1000
        
        for N in N_values:
            subset = self.digits[:N]
            counts = np.bincount(subset, minlength=self.base)
            probs = counts / N
            h_shannon.append(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
            
            for d in range(self.base):
                freq_data[d].append(probs[d] * 100)
                
            # Hydrodynamic Entropy
            if N > hydro_win:
                win_sub = self.digits[N-hydro_win : N]
                win_p = np.bincount(win_sub, minlength=self.base) / hydro_win
                h_hydro.append(-np.sum(win_p[win_p > 0] * np.log2(win_p[win_p > 0])))
            else:
                h_hydro.append(h_shannon[-1])

        # Find Saturation
        s_sat = self._find_sat(N_values, h_shannon, h_max)
        hy_sat = self._find_sat(N_values, h_hydro, h_max)

        # FFT Analysis
        sig = self.digits.astype(np.float32) - np.mean(self.digits)
        fft_res = np.abs(np.fft.rfft(sig))**2
        
        self.plot_results(N_values, h_shannon, h_hydro, h_max, s_sat, hy_sat, freq_data, fft_res)
        return (self.name, self.base, s_sat, hy_sat)

    def _find_sat(self, N_vals, H_vals, h_max):
        for i in range(len(H_vals) - CONFIRMATION_WINDOW):
            window = H_vals[i : i + CONFIRMATION_WINDOW]
            if np.all(np.abs(np.array(window) - h_max) < TOLERANCE):
                return N_vals[i]
        return None

    def plot_results(self, N, H_s, H_hy, h_max, s_sat, hy_sat, freq_data, fft):
        # 1. Entropy Comparison (Shannon & Hydrodynamic)
        plt.figure(figsize=(10, 5))
        plt.semilogx(N, H_s, label='Shannon (Global)')
        plt.semilogx(N, H_hy, label='Hydrodynamic (Local)', alpha=0.6)
        plt.axhline(h_max, color='r', linestyle='--')
        if s_sat: plt.axvline(s_sat, color='blue', ls=':', label=f'Shannon Sat @ {s_sat:,}')
        if hy_sat: plt.axvline(hy_sat, color='green', ls=':', label=f'Hydro Sat @ {hy_sat:,}')
        plt.title(f'Entropy: {self.name.upper()} (Base {self.base})')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'entropy_analysis.png'))
        plt.close()

        # 2. Digit Frequencies
        plt.figure(figsize=(10, 5))
        for d in range(self.base):
            plt.semilogx(N, freq_data[d], alpha=0.5)
        plt.axhline(100/self.base, color='k', ls='--')
        plt.title(f'Frequencies: {self.name.upper()} (Base {self.base})')
        plt.savefig(os.path.join(self.output_dir, 'frequencies.png'))
        plt.close()

        # 3. FFT Spectrum
        plt.figure(figsize=(10, 5))
        plt.loglog(fft, color='purple', alpha=0.7)
        plt.title(f'Spectral Intensity: {self.name.upper()}')
        plt.savefig(os.path.join(self.output_dir, 'spectrum.png'))
        plt.close()

def worker(args):
    try:
        analyzer = ComprehensiveAnalyzer(*args)
        return analyzer.run_analysis()
    except Exception as e:
        return (args[0], args[1], str(e), "Error")

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    print(f"System: {num_cores} cores. Analysis starting...")

    tasks = [(c, b, N_DIGITS) for c in CONSTANTS for b in BASES]

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(worker, tasks))

    print("\n" + "="*70)
    print(f"{'CONSTANT':<15} | {'BASE':<4} | {'SHANNON SAT':<15} | {'HYDRO SAT':<15}")
    print("-" * 70)
    for name, base, s_sat, hy_sat in results:
        s_str = f"{s_sat:,}" if isinstance(s_sat, int) else "N/A"
        h_str = f"{hy_sat:,}" if isinstance(hy_sat, int) else "N/A"
        print(f"{name:<15} | {base:<4} | {s_str:<15} | {h_str:<15}")
    print("="*70)