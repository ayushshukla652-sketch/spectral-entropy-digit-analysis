import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
import gc
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.tab10(np.linspace(0, 1, 10))

class EnhancedEntropyAnalyzer:
    
    def __init__(self, constant_name='sqrt2', n_digits=10_000_000):
        self.constant_name = constant_name
        self.n_digits = n_digits
        self.digits = None
        
    def generate_digits(self):
        print(f"Generating {self.n_digits:,} digits of {self.constant_name}...")
        
        # Set precision
        mp.dps = self.n_digits + 100
        
        # Compute constant
        if self.constant_name == 'pi':
            value = mp.pi
        elif self.constant_name == 'e':
            value = mp.e
        elif self.constant_name == 'sqrt2':
            value = mp.sqrt(2)
        elif self.constant_name == 'sqrt3':
            value = mp.sqrt(3)
        elif self.constant_name == 'sqrt5':
            value = mp.sqrt(5)
        elif self.constant_name == 'phi':
            value = (1 + mp.sqrt(5)) / 2
        elif self.constant_name == 'gamma':
            value = mp.euler
        else:
            raise ValueError(f"Unknown constant: {self.constant_name}")
        
        # Extract digits
        value_str = str(value).replace('.', '')
        self.digits = np.array([int(d) for d in value_str[:self.n_digits]], dtype=np.int8)
        
        print(f"Generated {len(self.digits):,} digits")
        print(f"First 50: {self.digits[:50]}")
        
        return self.digits
    
    def compute_digit_frequencies(self, N_samples=1000):
        """Compute digit frequencies at logarithmically spaced points"""
        N_values = np.logspace(1, np.log10(len(self.digits)), N_samples, dtype=int)
        N_values = np.unique(N_values)
        
        freq_dict = {d: [] for d in range(10)}
        
        for N in N_values:
            subset = self.digits[:N]
            counts = np.bincount(subset, minlength=10)
            freqs = counts / N * 100  # Percentage
            
            for d in range(10):
                freq_dict[d].append(freqs[d])
        
        return N_values, freq_dict
    
    def compute_shannon_entropy(self, N_samples=1000):
        """Compute Shannon entropy at multiple scales"""
        N_values = np.logspace(1, np.log10(len(self.digits)), N_samples, dtype=int)
        N_values = np.unique(N_values)
        
        shannon_entropies = []
        
        for N in N_values:
            subset = self.digits[:N]
            counts = np.bincount(subset, minlength=10)
            probs = counts / N
            probs = probs[probs > 0]
            H = -np.sum(probs * np.log2(probs))
            shannon_entropies.append(H)
        
        return N_values, np.array(shannon_entropies)
    
    def compute_spectral_entropy(self, N_samples=100):
        """Compute spectral entropy at multiple scales"""
        N_values = np.logspace(3, np.log10(len(self.digits)), N_samples, dtype=int)
        N_values = np.unique(N_values)
        
        spectral_entropies = []
        
        for i, N in enumerate(N_values):
            if i % 10 == 0:
                print(f"  Spectral entropy: {i}/{len(N_values)}, N={N:,}")
            
            subset = self.digits[:N].astype(np.float32)
            
            # Center
            subset = subset - np.mean(subset)
            
            # FFT
            psi = np.fft.fft(subset)
            
            # Modal energy (first N/2 modes)
            E_k = 0.5 * np.abs(psi[:N//2])**2
            
            # Normalize
            p_k = E_k / np.sum(E_k)
            p_k = p_k[p_k > 0]
            
            # Spectral entropy
            S_H = -np.sum(p_k * np.log2(p_k))
            spectral_entropies.append(S_H)
            
            # Clean up
            del subset, psi, E_k, p_k
            gc.collect()
        
        return N_values, np.array(spectral_entropies)
    
    def compute_global_spectrum(self):
        """Compute global Fourier spectrum"""
        print(f"Computing global spectrum for N={len(self.digits):,}...")
        
        # Center
        digits_centered = self.digits.astype(np.float32) - np.mean(self.digits)
        
        # FFT
        psi = np.fft.fft(digits_centered)
        
        # Power spectrum
        N = len(self.digits)
        E_k = 0.5 * np.abs(psi[:N//2])**2
        
        return np.arange(1, N//2 + 1), E_k
    
    def compute_digit_resolved_spectra(self):
        """Compute spectra for each digit separately"""
        print("Computing digit-resolved spectra...")
        
        spectra = {}
        N = len(self.digits)
        
        for digit in range(10):
            # Create binary sequence: 1 where digit appears, 0 otherwise
            binary_seq = (self.digits == digit).astype(np.float32)
            
            # Center
            binary_seq = binary_seq - np.mean(binary_seq)
            
            # FFT
            psi = np.fft.fft(binary_seq)
            
            # Power
            E_k = 0.5 * np.abs(psi[:N//2])**2
            
            spectra[digit] = E_k
        
        wavenumbers = np.arange(1, N//2 + 1)
        
        return wavenumbers, spectra
    
    def find_saturation_point(self, N_values, H_values, threshold=0.001, confirmation_window=20):
 
        H_max = np.log2(10)
        
        for i, (N, H) in enumerate(zip(N_values, H_values)):
            # Check if current point is within threshold
            if abs(H - H_max) < threshold:
                # Check if next 'confirmation_window' points also satisfy criterion
                if i < len(N_values) - confirmation_window:
                    remaining = H_values[i:i+confirmation_window]
                    
                    # All points in window must be within threshold
                    if np.all(np.abs(remaining - H_max) < threshold):
                        print(f"   Saturation confirmed with {confirmation_window}-point window")
                        print(f"   Range checked: N=[{N:,} to {N_values[i+confirmation_window-1]:,}]")
                        return N
        
        print(f"   Warning: No saturation point found with {confirmation_window}-point confirmation")
        return None
    
    def find_logarithmic_onset(self, N_values, S_values, min_N=1000, 
                              R2_threshold=0.995, confirmation_window=20):

        # Fit logarithmic model to subsets and check R²
        mask = N_values >= min_N
        N_fit = N_values[mask]
        S_fit = S_values[mask]
        
        best_R2 = 0
        best_N_onset = None
        best_alpha = None
        
        # Need at least confirmation_window points for the fit
        for i in range(10, len(N_fit) - confirmation_window):
            N_subset = N_fit[i:]
            S_subset = S_fit[i:]
            
            # Only consider if we have enough points
            if len(N_subset) < confirmation_window:
                continue
            
            # Fit
            log_N = np.log10(N_subset)
            coeffs = np.polyfit(log_N, S_subset, 1)
            S_pred = coeffs[0] * log_N + coeffs[1]
            
            # R²
            SS_res = np.sum((S_subset - S_pred)**2)
            SS_tot = np.sum((S_subset - np.mean(S_subset))**2)
            R2 = 1 - SS_res / SS_tot
            
            if R2 > R2_threshold and R2 > best_R2:
                best_R2 = R2
                best_N_onset = N_subset[0]
                best_alpha = coeffs[0]
        
        if best_N_onset:
            print(f"   Logarithmic onset confirmed with R²={best_R2:.6f}")
            print(f"   Fitted range: N=[{best_N_onset:,} to {N_fit[-1]:,}]")
            print(f"   Number of points in fit: {len(N_fit[N_fit >= best_N_onset])}")
        else:
            print(f"   Warning: No logarithmic onset found with R²>{R2_threshold}")
        
        return best_N_onset
    
    def plot_digit_frequencies(self, N_values, freq_dict, output_dir='entropy_plots'):
        """Plot digit frequency convergence (split 0-4 and 5-9)"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Digits 0-4
        ax = axes[0]
        for d in range(5):
            ax.semilogx(N_values, freq_dict[d], label=f'Digit {d}', 
                       color=colors[d], alpha=0.8, linewidth=1.5)
        ax.axhline(10, color='black', linestyle='--', linewidth=2, label='Uniform (10%)')
        ax.set_xlabel('Number of digits (N)', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'{self.constant_name} Digit Frequency Distribution (0-4)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 30])
        
        # Digits 5-9
        ax = axes[1]
        for d in range(5, 10):
            ax.semilogx(N_values, freq_dict[d], label=f'Digit {d}', 
                       color=colors[d], alpha=0.8, linewidth=1.5)
        ax.axhline(10, color='black', linestyle='--', linewidth=2, label='Uniform (10%)')
        ax.set_xlabel('Number of digits (N)', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'{self.constant_name} Digit Frequency Distribution (5-9)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 30])
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.constant_name}_digit_frequencies.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_shannon_entropy_with_marker(self, N_values, H_values, output_dir='entropy_plots'):
        """Plot Shannon entropy with saturation marker"""
        
        # Find saturation point with 20-point confirmation
        N_sat = self.find_saturation_point(N_values, H_values, 
                                          threshold=0.01, 
                                          confirmation_window=20)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogx(N_values, H_values, 'b-', linewidth=2.5, label=self.constant_name)
        ax.axhline(np.log2(10), color='red', linestyle='--', linewidth=2, 
                  label=r'Maximum ($\log_2 10 \approx 3.322$)')
        
        # Mark saturation point
        if N_sat:
            ax.axvline(N_sat, color='green', linestyle=':', linewidth=2, alpha=0.7,
                      label=f'Saturation at N={N_sat:,}')
            ax.plot(N_sat, np.log2(10), 'go', markersize=12, markeredgecolor='darkgreen', 
                   markeredgewidth=2, zorder=5)
            
            # Add annotation
            ax.annotate(f'Saturates at\nN ≈ {N_sat:.2e}\n(20-pt confirmed)', 
                       xy=(N_sat, np.log2(10)), 
                       xytext=(N_sat * 10, np.log2(10) - 0.3),
                       fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        ax.set_xlabel('Number of digits (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Shannon Entropy (bits)', fontsize=13, fontweight='bold')
        ax.set_title(f'Shannon Entropy Convergence: {self.constant_name}', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim([2.7, 3.4])
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.constant_name}_shannon_entropy.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
        
        return N_sat
    
    def plot_spectral_entropy_with_marker(self, N_values, S_values, output_dir='entropy_plots'):
        """Plot spectral entropy with logarithmic onset marker"""
        
        # Find logarithmic onset with 20-point confirmation
        N_onset = self.find_logarithmic_onset(N_values, S_values, 
                                             min_N=1000,
                                             R2_threshold=0.99,
                                             confirmation_window=20)
        
        # Fit for entire range N > 1000
        mask = N_values >= 1000
        log_N = np.log10(N_values[mask])
        coeffs = np.polyfit(log_N, S_values[mask], 1)
        alpha, beta = coeffs
        
        # R²
        S_pred = alpha * log_N + beta
        SS_res = np.sum((S_values[mask] - S_pred)**2)
        SS_tot = np.sum((S_values[mask] - np.mean(S_values[mask]))**2)
        R2 = 1 - SS_res / SS_tot
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data
        ax.loglog(N_values, S_values, 'b-', linewidth=2.5, label=f'{self.constant_name} (data)')
        
        # Plot fit
        N_fit_range = np.logspace(3, np.log10(N_values[-1]), 100)
        S_fit_range = alpha * np.log10(N_fit_range) + beta
        ax.loglog(N_fit_range, S_fit_range, 'r--', linewidth=2, alpha=0.7,
                 label=f'Fit: α={alpha:.2f}, R²={R2:.4f}')
        
        # Mark onset
        if N_onset:
            ax.axvline(N_onset, color='green', linestyle=':', linewidth=2, alpha=0.7,
                      label=f'Logarithmic onset N={N_onset:,}')
            S_onset = alpha * np.log10(N_onset) + beta
            ax.plot(N_onset, S_onset, 'go', markersize=12, markeredgecolor='darkgreen',
                   markeredgewidth=2, zorder=5)
            
            # Annotation
            ax.annotate(f'Logarithmic at\nN ≈ {N_onset:.2e}\n(20-pt confirmed)', 
                       xy=(N_onset, S_onset), 
                       xytext=(N_onset * 0.1, S_onset * 1.5),
                       fontsize=11, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
        
        # Add theoretical prediction
        alpha_theory = np.log(10) / np.log(2)
        N_theory_range = np.logspace(3, np.log10(N_values[-1]), 100)
        S_theory = alpha_theory * np.log10(N_theory_range) - 1
        ax.loglog(N_theory_range, S_theory, 'k:', linewidth=2, alpha=0.5,
                 label=f'Theory: α={alpha_theory:.2f}')
        
        ax.set_xlabel('Number of digits (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Spectral Entropy (bits)', fontsize=13, fontweight='bold')
        ax.set_title(f'Spectral Entropy Scaling: {self.constant_name}', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.constant_name}_spectral_entropy.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
        
        return N_onset, alpha, R2
    
    def plot_global_spectrum(self, wavenumbers, spectrum, output_dir='entropy_plots'):
        """Plot global Fourier spectrum"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(wavenumbers, spectrum, 'b-', linewidth=1, alpha=0.7)
        ax.set_xlabel('Wavenumber k', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Spectral Intensity $|E_k|^2$', fontsize=13, fontweight='bold')
        ax.set_title(f'Global Fourier Spectrum: {self.constant_name} (N={len(self.digits):,})', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.constant_name}_global_spectrum.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def plot_digit_resolved_spectra(self, wavenumbers, spectra, output_dir='entropy_plots'):
        """Plot digit-resolved spectra"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for digit in range(10):
            ax.loglog(wavenumbers, spectra[digit], label=f'Digit {digit}', 
                     color=colors[digit], alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Wavenumber k', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Spectral Intensity $|E_k|^2$', fontsize=13, fontweight='bold')
        ax.set_title(f'Digit-Resolved Fourier Spectra: {self.constant_name}', 
                    fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f'{self.constant_name}_digit_resolved_spectrum.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    def full_analysis(self, output_dir='entropy_plots'):
        """Run complete analysis pipeline"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print(f"FULL ANALYSIS: {self.constant_name.upper()}")
        print(f"Digits: {self.n_digits:,}")
        print(f"Output directory: {output_dir}/")
        print("="*70)
        
        # Generate digits if not already done
        if self.digits is None:
            self.generate_digits()
        
        # Save digits
        digits_file = os.path.join(output_dir, f'{self.constant_name}_digits.npy')
        np.save(digits_file, self.digits)
        print(f"\n✓ Saved digits: {digits_file}")
        
        # 1. Digit frequencies
        print("\n" + "-"*70)
        print("1. Computing digit frequencies...")
        print("-"*70)
        N_freq, freq_dict = self.compute_digit_frequencies()
        self.plot_digit_frequencies(N_freq, freq_dict, output_dir)
        
        # 2. Shannon entropy
        print("\n" + "-"*70)
        print("2. Computing Shannon entropy...")
        print("-"*70)
        N_shannon, H_shannon = self.compute_shannon_entropy()
        N_sat = self.plot_shannon_entropy_with_marker(N_shannon, H_shannon, output_dir)
        
        if N_sat:
            print(f"\n   ✓ Shannon saturation: N = {N_sat:,}")
        else:
            print(f"\n   ✗ No saturation found with 20-point confirmation")
        
        # Save Shannon data
        shannon_file = os.path.join(output_dir, f'{self.constant_name}_shannon_data.npz')
        np.savez(shannon_file, N=N_shannon, H=H_shannon, N_sat=N_sat)
        print(f"  ✓ Saved data: {shannon_file}")
        
        # 3. Spectral entropy
        print("\n" + "-"*70)
        print("3. Computing spectral entropy...")
        print("-"*70)
        N_spectral, S_spectral = self.compute_spectral_entropy()
        N_onset, alpha, R2 = self.plot_spectral_entropy_with_marker(N_spectral, S_spectral, output_dir)
        
        if N_onset:
            print(f"\n   ✓ Logarithmic onset: N = {N_onset:,}")
        else:
            print(f"\n   ✗ No onset found with 20-point confirmation")
        
        print(f"   ✓ Scaling exponent: α = {alpha:.2f}")
        print(f"   ✓ Goodness of fit: R² = {R2:.6f}")
        
        # Compare with theory
        alpha_theory = np.log(10) / np.log(2)
        diff_pct = abs(alpha - alpha_theory) / alpha_theory * 100
        print(f"\n   Theoretical α = {alpha_theory:.2f}")
        print(f"   Difference: {diff_pct:.1f}%")
        
        # Save spectral data
        spectral_file = os.path.join(output_dir, f'{self.constant_name}_spectral_data.npz')
        np.savez(spectral_file, N=N_spectral, S_H=S_spectral, alpha=alpha, R2=R2, N_onset=N_onset)
        print(f"  ✓ Saved data: {spectral_file}")
        
        # 4. Global spectrum
        print("\n" + "-"*70)
        print("4. Computing global spectrum...")
        print("-"*70)
        wavenumbers, spectrum = self.compute_global_spectrum()
        self.plot_global_spectrum(wavenumbers, spectrum, output_dir)
        
        # 5. Digit-resolved spectra
        print("\n" + "-"*70)
        print("5. Computing digit-resolved spectra...")
        print("-"*70)
        wavenumbers_dr, spectra_dr = self.compute_digit_resolved_spectra()
        self.plot_digit_resolved_spectra(wavenumbers_dr, spectra_dr, output_dir)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"All files saved in: {output_dir}/")
        print("="*70)
        
        # List all saved files for this constant
        print(f"\nFiles for {self.constant_name}:")
        all_files = sorted(os.listdir(output_dir))
        constant_files = [f for f in all_files if f.startswith(self.constant_name)]
        for file in constant_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"  ✓ {file:50s} ({file_size:.2f} MB)")
        
        # Return summary
        return {
            'constant': self.constant_name,
            'n_digits': self.n_digits,
            'N_sat': N_sat,
            'N_onset': N_onset,
            'alpha': alpha,
            'alpha_theory': alpha_theory,
            'R2': R2,
            'N_shannon': N_shannon,
            'H_shannon': H_shannon,
            'N_spectral': N_spectral,
            'S_spectral': S_spectral
        }

if __name__ == "__main__":
    
    # All plots will be saved in entropy_plots/ folder
    
    print("="*70)
    print("STARTING ENTROPY ANALYSIS - 10 MILLION DIGITS")
    print("Saturation confirmation: 20-point window")
    print("All plots will be saved in: entropy_plots/")
    print("="*70)
    
    # Analyze √2
    print("\n\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ANALYZING √2".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    analyzer_sqrt2 = EnhancedEntropyAnalyzer('sqrt2', n_digits=10_000_000)
    results_sqrt2 = analyzer_sqrt2.full_analysis(output_dir='entropy_plots')
    
    # Analyze π
    print("\n\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ANALYZING π".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    analyzer_pi = EnhancedEntropyAnalyzer('pi', n_digits=10_000_000)
    results_pi = analyzer_pi.full_analysis(output_dir='entropy_plots')
    
    # Analyze e
    print("\n\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ANALYZING e".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    analyzer_e = EnhancedEntropyAnalyzer('e', n_digits=10_000_000)
    results_e = analyzer_e.full_analysis(output_dir='entropy_plots')
    
    # Final summary
    print("\n\n")
    print("="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print("\nAll plots and data saved in: entropy_plots/")
    print("\nTotal files created:")
    all_files = os.listdir('entropy_plots')
    total_size = sum(os.path.getsize(os.path.join('entropy_plots', f)) for f in all_files) / (1024**2)
    print(f"  - {len(all_files)} files ({total_size:.1f} MB)")
    print(f"\nBreakdown:")
    print(f"  - √2: {len([f for f in all_files if f.startswith('sqrt2')])} files")
    print(f"  - π:  {len([f for f in all_files if f.startswith('pi')])} files")
    print(f"  - e:  {len([f for f in all_files if f.startswith('e')])} files")
    
    # Summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\n{:<10} {:<15} {:<15} {:<8} {:<8}".format(
        "Constant", "Shannon Sat.", "Spectral Onset", "α", "R²"))
    print("-" * 70)
    
    for results in [results_sqrt2, results_pi, results_e]:
        const = results['constant']
        n_sat = results['N_sat'] if results['N_sat'] else 0
        n_onset = results['N_onset'] if results['N_onset'] else 0
        
        print("{:<10} {:<15,} {:<15,} {:<8.2f} {:<8.6f}".format(
            const, n_sat, n_onset, results['alpha'], results['R2']))
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"  Theoretical α = {results_sqrt2['alpha_theory']:.2f}")
    print(f"  √2 measured α = {results_sqrt2['alpha']:.2f} "
          f"(diff: {abs(results_sqrt2['alpha']-results_sqrt2['alpha_theory'])/results_sqrt2['alpha_theory']*100:.1f}%)")
    print(f"  π measured α  = {results_pi['alpha']:.2f} "
          f"(diff: {abs(results_pi['alpha']-results_pi['alpha_theory'])/results_pi['alpha_theory']*100:.1f}%)")
    print(f"  e measured α  = {results_e['alpha']:.2f} "
          f"(diff: {abs(results_e['alpha']-results_e['alpha_theory'])/results_e['alpha_theory']*100:.1f}%)")
    
    print("\n" + "="*70)