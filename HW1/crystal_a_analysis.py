"""
Independent analysis of Crystal A powder diffraction data.

Goal: From scattering angles (alpha = 2*theta), determine:
  1. The Bravais lattice type (SC, BCC, FCC, Diamond)
  2. The lattice parameter 'a'

Given:
  - k_w = 8.8581 Ang^{-1}  (wave vector)
  - lambda = 2*pi / k_w
  - Bragg's Law: lambda = 2*d*sin(theta)
  - For cubic: 1/d^2 = (h^2 + k^2 + l^2) / a^2

Method:
  sin^2(theta) = (lambda^2 / 4a^2) * (h^2 + k^2 + l^2)
  
  So ratios of sin^2(theta) values should give ratios of (h^2+k^2+l^2).
  The pattern of allowed (h^2+k^2+l^2) values identifies the lattice type:
    SC:      1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, ...
    BCC:     2, 4, 6, 8, 10, 12, 14, 16, ...   (h+k+l = even)
    FCC:     3, 4, 8, 11, 12, 16, 19, 20, ...  (h,k,l all odd or all even)
    Diamond: 3, 8, 11, 16, 19, 24, ...          (FCC + additional extinctions)
"""

import numpy as np
import pandas as pd
import itertools

# ============================================================
# Constants
# ============================================================
kw = 8.8581  # wave vector in Ang^{-1}
lam = 2 * np.pi / kw  # wavelength in Ang
print(f"Wavelength: lambda = 2*pi / {kw} = {lam:.6f} Ang")
print(f"  (This is very close to Mo K-alpha = 0.7107 Ang)")
print()

# ============================================================
# Read data
# ============================================================
df = pd.read_csv("data.csv")
alpha = df['A'].values  # alpha = 2*theta, in radians
theta = alpha / 2.0     # Bragg angle theta

print("="*70)
print("RAW DATA")
print("="*70)
print(f"{'Peak':>5} {'alpha (rad)':>12} {'theta (rad)':>12} {'theta (deg)':>12}")
for i, (a, t) in enumerate(zip(alpha, theta)):
    print(f"{i+1:5d} {a:12.6f} {t:12.6f} {np.degrees(t):12.4f}")
print()

# ============================================================
# Step 1: Compute sin^2(theta) for each peak
# ============================================================
sin2 = np.sin(theta)**2

print("="*70)
print("STEP 1: sin^2(theta) values")
print("="*70)
for i, s in enumerate(sin2):
    print(f"  Peak {i+1:2d}: sin^2(theta) = {s:.6f}")
print()

# ============================================================
# Step 2: Normalize by dividing by the smallest sin^2(theta)
# ============================================================
ratios = sin2 / sin2[0]

print("="*70)
print("STEP 2: Ratios sin^2(theta_i) / sin^2(theta_1)")
print("="*70)
for i, r in enumerate(ratios):
    print(f"  Peak {i+1:2d}: ratio = {r:.6f}  (nearest int = {round(r)})")
print()

# Check if all ratios are close to integers
residuals = np.abs(ratios - np.round(ratios))
print(f"  Max deviation from integer: {np.max(residuals):.6f}")
if np.max(residuals) < 0.05:
    print("  [OK] All ratios are very close to integers => consistent with CUBIC system")
else:
    print("  [!!] Some ratios deviate significantly from integers")
print()

# ============================================================
# Step 3: Identify the sequence of N = h^2 + k^2 + l^2
# ============================================================
N_values = np.round(ratios).astype(int)

print("="*70)
print("STEP 3: Sequence of N = h^2 + k^2 + l^2")
print("="*70)
print(f"  Observed N values: {list(N_values)}")
print()

# Known sequences for cubic lattices
sc_sequence  = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18]
bcc_sequence = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34]
fcc_sequence = [3, 4, 8, 11, 12, 16, 19, 20, 24, 27, 32, 35, 36, 40, 43, 44]
diamond_seq  = [3, 8, 11, 16, 19, 24, 27, 32, 35, 40, 43, 48, 51, 56, 59, 64]

print("  Reference sequences (first 15 peaks):")
print(f"    SC:      {sc_sequence[:15]}")
print(f"    BCC:     {bcc_sequence[:15]}")
print(f"    FCC:     {fcc_sequence[:15]}")
print(f"    Diamond: {diamond_seq[:15]}")
print()

# Check which sequence matches
n_peaks = len(N_values)
matches = {}
for name, seq in [("SC", sc_sequence), ("BCC", bcc_sequence), ("FCC", fcc_sequence), ("Diamond", diamond_seq)]:
    if n_peaks <= len(seq):
        if list(N_values) == seq[:n_peaks]:
            matches[name] = seq
            
print("  Pattern matching results:")
for name in ["SC", "BCC", "FCC", "Diamond"]:
    seq = {"SC": sc_sequence, "BCC": bcc_sequence, "FCC": fcc_sequence, "Diamond": diamond_seq}[name]
    match = list(N_values) == seq[:n_peaks]
    print(f"    {name:8s}: {'>> MATCH <<' if match else 'no match'}")
    if not match:
        # Show where it diverges
        for j in range(min(n_peaks, len(seq))):
            if N_values[j] != seq[j]:
                print(f"             (First mismatch at peak {j+1}: observed N={N_values[j]}, expected N={seq[j]})")
                break
print()

# ============================================================
# Step 4: Find Miller indices for each peak
# ============================================================
def find_hkl(target):
    """Find all (h,k,l) with h^2+k^2+l^2 = target, h,k,l >= 0."""
    target = int(target)
    max_val = int(np.sqrt(target)) + 1
    solutions = []
    for h, k, l in itertools.product(range(0, max_val + 1), repeat=3):
        if h**2 + k**2 + l**2 == target:
            solutions.append((h, k, l))
    return sorted(set(solutions), reverse=True)

print("="*70)
print("STEP 4: Miller indices for each peak")
print("="*70)
for i, N in enumerate(N_values):
    hkl_list = find_hkl(N)
    # Show the "canonical" (highest symmetry) index first
    if hkl_list:
        print(f"  Peak {i+1:2d}: N = {N:2d}  =>  {hkl_list}")
    else:
        print(f"  Peak {i+1:2d}: N = {N:2d}  =>  NO SOLUTION (forbidden by number theory)")
print()

# ============================================================
# Step 5: Compute lattice parameter 'a' from each peak
# ============================================================
# From Bragg's law: d = lambda / (2 sin(theta))
# For cubic: a = d * sqrt(h^2 + k^2 + l^2) = d * sqrt(N)
# Therefore: a = lambda * sqrt(N) / (2 * sin(theta))
#          = pi * sqrt(N) / (kw * sin(theta))

print("="*70)
print("STEP 5: Lattice parameter from each peak")
print("="*70)

a_values = []
for i, (t, N) in enumerate(zip(theta, N_values)):
    hkl_list = find_hkl(N)
    if not hkl_list:
        print(f"  Peak {i+1:2d}: N = {N:2d}  =>  SKIPPED (no valid hkl)")
        continue
    
    d = lam / (2 * np.sin(t))
    a = d * np.sqrt(N)
    a_values.append(a)
    
    # Also compute using pi/kw formulation as a cross-check
    a_check = np.pi * np.sqrt(N) / (kw * np.sin(t))
    
    hkl = hkl_list[0]  # use highest-index permutation
    print(f"  Peak {i+1:2d}: N = {N:2d}  hkl = {hkl}  d = {d:.6f} Ang  a = {a:.6f} Ang  (check: {a_check:.6f})")

a_values = np.array(a_values)
a_mean = np.mean(a_values)
a_std = np.std(a_values)

print()
print(f"  Mean lattice parameter:  a = {a_mean:.6f} ± {a_std:.6f} Ang")
print(f"  Spread (max - min):      {np.max(a_values) - np.min(a_values):.6f} Ang")
print()

# ============================================================
# Step 6: Summary and crystal identification
# ============================================================
print("="*70)
print("SUMMARY")
print("="*70)
print(f"  Wavelength:              lam = {lam:.6f} Ang")
print(f"  Number of peaks:         {n_peaks}")
print(f"  N sequence:              {list(N_values)}")
print()

if matches:
    lattice_type = list(matches.keys())[0]
    print(f"  Lattice type:            {lattice_type}")
else:
    print(f"  Lattice type:            UNKNOWN (no standard sequence matched)")
    
print(f"  Lattice parameter:       a = {a_mean:.4f} ± {a_std:.4f} Ang")
print()

# Cross-check: what element could this be?
print("  Cross-reference with known lattice parameters:")
print(f"    Polonium (SC):     a = 3.345 Ang")
print(f"    Iron (BCC):        a = 2.870 Ang")  
print(f"    Tungsten (BCC):    a = 3.165 Ang")
print(f"    Copper (FCC):      a = 3.615 Ang")
print(f"    Aluminum (FCC):    a = 4.050 Ang")
print(f"    NaCl (FCC):        a = 5.640 Ang")
print()

# ============================================================
# Step 7: Verify your notebook results
# ============================================================
print("="*70)
print("VERIFICATION OF YOUR NOTEBOOK")
print("="*70)

# Check the formula you used: (1/(kw * 2 * sin(theta))) * sqrt(N)
# vs correct: (pi / (kw * sin(theta))) * sqrt(N)
# Ratio = pi / (1/2) = 2*pi
# So your formula was off by a factor of 2*pi

your_a = (1 / (kw * 2 * np.sin(theta[0]))) * np.sqrt(N_values[0])
correct_a = (np.pi / (kw * np.sin(theta[0]))) * np.sqrt(N_values[0])

print(f"  Your formula gives:      a = {your_a:.6f} Ang")
print(f"  Correct formula gives:   a = {correct_a:.6f} Ang")
print(f"  Ratio (correct/yours):   {correct_a/your_a:.6f}")
print(f"  Expected ratio (2*pi):   {2*np.pi:.6f}")
print()

if abs(correct_a/your_a - 2*np.pi) < 0.01:
    print("  WARNING: Your get_cubic_a() formula is missing a factor of 2*pi.")
    print("    You used:    a = sqrt(N) / (kw * 2 * sin(theta))")
    print("    Should be:   a = pi * sqrt(N) / (kw * sin(theta))")
    print("    Or equiv:    a = lambda * sqrt(N) / (2 * sin(theta))")
