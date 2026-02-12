import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
from multiprocessing import Pool
import os
import math

# ============================================================
# SETTINGS
# ============================================================

N_total = 1_000_000
num_samples = 1000
N_CORES = 40
SAVE_DIR = "hpc_full_plots"

os.makedirs(SAVE_DIR, exist_ok=True)
np.random.seed(0)

# ============================================================
# GENERATORS
# ============================================================

def generate_random(N):
    return np.random.randint(0,10,N)

def generate_periodic(N):
    return np.array([i % 10 for i in range(N)], dtype=int)

def generate_champernowne(N):
    s=""
    i=1
    while len(s)<N:
        s+=str(i)
        i+=1
    return np.array(list(s[:N]),dtype=int)

def generate_logistic(N):
    x=0.123456
    digits=np.zeros(N,dtype=int)
    for i in range(N):
        x=4*x*(1-x)
        digits[i]=int(x*10)%10
    return digits

def generate_lcg(N,a=1664525,c=1013904223,m=2**32):
    x=1
    digits=np.zeros(N,dtype=int)
    for i in range(N):
        x=(a*x+c)%m
        digits[i]=x%10
    return digits

def generate_thue(N):
    digits=np.zeros(N,dtype=int)
    for i in range(N):
        digits[i]=(bin(i).count("1")%2)*5
    return digits

def debruijn(k,n):
    a=[0]*(k*n)
    seq=[]
    def db(t,p):
        if t>n:
            if n%p==0:
                seq.extend(a[1:p+1])
        else:
            a[t]=a[t-p]
            db(t+1,p)
            for j in range(a[t-p]+1,k):
                a[t]=j
                db(t+1,t)
    db(1,1)
    return np.array(seq,dtype=int)

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
    return -np.sum(p*np.log2(p))

def worker(args):
    digits, N = args
    return spectral_entropy(digits[:N])

def compute_scaling_parallel(digits):
    N_vals = np.unique(
        np.logspace(3,np.log10(len(digits)),num_samples).astype(int)
    )
    tasks=[(digits,N) for N in N_vals]
    with Pool(N_CORES) as pool:
        S_vals=pool.map(worker,tasks)
    return np.log10(N_vals), np.array(S_vals)

# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":

    sequences={
        "Random":generate_random(N_total),
        "Periodic":generate_periodic(N_total),
        "Champernowne":generate_champernowne(N_total),
        "Logistic":generate_logistic(N_total),
        "LCG":generate_lcg(N_total),
        "Thue-Morse":generate_thue(N_total),
        "DeBruijn":debruijn(10,7)[:N_total]
    }

    for name,digits in sequences.items():

        print(f"Processing {name} using {N_CORES} cores...")

        logN,S_vals=compute_scaling_parallel(digits)

        plt.figure(figsize=(7,6))
        plt.plot(logN,S_vals,linewidth=1.5,label="Data")

        xfit=np.linspace(logN.min(),logN.max(),500)

        if name=="DeBruijn":
            # Quadratic fit
            coeffs=np.polyfit(logN,S_vals,2)
            a,b,c=coeffs
            yfit=a*xfit**2+b*xfit+c
            plt.plot(xfit,yfit,linewidth=3,
                     label="Quadratic fit")

            # alpha_eff
            alpha_eff=2*a*xfit+b
            plt.figure(figsize=(7,6))
            plt.plot(xfit,alpha_eff,linewidth=2)
            plt.axhline(3.322,linestyle="--")
            plt.xlabel("log10(N)")
            plt.ylabel("α_eff(N)")
            plt.title("DeBruijn Effective Exponent")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}/DeBruijn_alpha_eff.png",dpi=300)
            plt.close()

        else:
            # Linear fit
            coeffs=np.polyfit(logN,S_vals,1)
            alpha,beta=coeffs
            yfit=alpha*xfit+beta
            plt.plot(xfit,yfit,linewidth=3,
                     label=f"Fit α={alpha:.3f}")

        # Theory line
        plt.plot(xfit,3.322*xfit-1,'k--',
                 label="Theory α=3.322")

        plt.xlabel("log10(N)")
        plt.ylabel("Spectral Entropy $S_H$")
        plt.title(f"{name} Spectral Scaling")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/{name}_spectral.png",dpi=300)
        plt.close()

    print("All 10-sequence HPC spectral plots completed.")
