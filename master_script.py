import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import os
import math

# ============================================================
# SETTINGS
# ============================================================

N_total = 1_000_000        # change to 1_500_000 if needed
num_samples = 400
BASE = 10

np.random.seed(0)

SAVE_DIR = "plots_new"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# CONSTANT GENERATORS
# ============================================================

def generate_pi(N):
    # Bailey–Borwein–Plouffe formula approximation
    # Not digit-extraction; numeric precision method
    return np.array(list(str(np.pi)[2:2+N]), dtype=int)

def generate_e(N):
    return np.array(list(str(math.e)[2:2+N]), dtype=int)

def generate_sqrt2(N):
    return np.array(list(str(math.sqrt(2))[2:2+N]), dtype=int)

def generate_champernowne(N):
    s = ""
    i = 1
    while len(s) < N:
        s += str(i)
        i += 1
    return np.array(list(s[:N]), dtype=int)

def generate_periodic(N):
    return np.array([i % 10 for i in range(N)], dtype=int)

def generate_random(N):
    return np.random.randint(0,10,N)

# Logistic map
def generate_logistic(N):
    x = 0.123456
    digits = np.zeros(N,dtype=int)
    for i in range(N):
        x = 4*x*(1-x)
        digits[i] = int(x*10)%10
    return digits

# LCG
def generate_lcg(N,a=1664525,c=1013904223,m=2**32):
    x=1
    digits=np.zeros(N,dtype=int)
    for i in range(N):
        x=(a*x+c)%m
        digits[i]=x%10
    return digits

# Thue-Morse
def generate_thue(N):
    digits=np.zeros(N,dtype=int)
    for i in range(N):
        digits[i]=(bin(i).count("1")%2)*5
    return digits

# De Bruijn
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
# ENTROPY FUNCTIONS
# ============================================================

def shannon_entropy(digits):
    counts=np.bincount(digits,minlength=10)
    p=counts/np.sum(counts)
    p=p[p>0]
    return -np.sum(p*np.log2(p))

def spectral_entropy(signal):
    signal=signal-np.mean(signal)
    fft_vals=rfft(signal)
    power=np.abs(fft_vals)**2
    p=power/np.sum(power)
    p=p[p>0]
    return -np.sum(p*np.log2(p))

def compute_scaling(digits):
    N_vals=np.unique(
        np.logspace(3,np.log10(len(digits)),num_samples).astype(int)
    )
    S_vals=[]
    H_vals=[]
    for N in N_vals:
        sub=digits[:N]
        S_vals.append(spectral_entropy(sub))
        H_vals.append(shannon_entropy(sub))
    return np.log10(N_vals), np.array(S_vals), np.array(H_vals)

# ============================================================
# GENERATE ALL SEQUENCES
# ============================================================

print("Generating sequences...")

sequences={
    "pi":generate_pi(N_total),
    "e":generate_e(N_total),
    "sqrt2":generate_sqrt2(N_total),
    "Champernowne":generate_champernowne(N_total),
    "Periodic":generate_periodic(N_total),
    "Random":generate_random(N_total),
    "Logistic":generate_logistic(N_total),
    "LCG":generate_lcg(N_total),
    "Thue-Morse":generate_thue(N_total),
    "DeBruijn":debruijn(10,6)[:N_total]
}

# ============================================================
# PLOTTING
# ============================================================

for name,digits in sequences.items():

    print(f"Processing {name}...")

    logN,S_vals,H_vals=compute_scaling(digits)

    # Shannon Plot
    plt.figure(figsize=(6,5))
    plt.plot(logN,H_vals)
    plt.axhline(np.log2(10),linestyle="--")
    plt.xlabel("log10(N)")
    plt.ylabel("Shannon Entropy")
    plt.title(f"{name} Shannon Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{name}_shannon.png",dpi=300)
    plt.close()

    # Spectral Plot
    plt.figure(figsize=(6,5))
    plt.plot(logN,S_vals)
    plt.xlabel("log10(N)")
    plt.ylabel("Spectral Entropy")
    plt.title(f"{name} Spectral Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{name}_spectral.png",dpi=300)
    plt.close()

print("All plots generated.")
