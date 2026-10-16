"""
generate_all_figures.py: Complete generator for all Homework 2 electronic band structure figures.

Uses strictly the user's inverse eta-bar formulation:
    M(k) = eps_bar + t_bar(k) @ inv(eta_bar(k))
    M c' = E c'   (standardized eigenvalue problem)
    evals = np.linalg.eigvals(M)
    E(k) = np.sort(np.real(evals))

Generates figures for both:
1. Crystal A: Square Lattice (two orbitals per site, 2x2 secular problem)
   Path: Gamma (0,0) -> X (pi/a, 0) -> M (pi/a, pi/a) -> Gamma (0,0)
2. Crystal B: Centered Rectangular Lattice (two-site conventional basis, 3x3 secular problem)
   Path: Gamma (0,0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0,0) with b = 1.2 a

For each lattice parameter in the problem statement:
   a = 5.0 * d0, a = 7.0 * d0, a = 9.0 * d0

And for all physical approximations:
   - Full Model (NN + NNN with Overlap)
   - Neglecting Overlaps (No Overlap: eta_nm(R) = delta_nm * delta_R,0, S = Identity)
   - Neglecting Next-Nearest Neighbors (NN Only, with Overlap)
   - Neglecting Nearest Neighbors (NNN Only, with Overlap)

Outputs are saved in both:
- HW2/plots/
- HW2/Writeup/figures/
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# Model Parameters and Matrix Calculations
# ==============================================================================

def calc_eta_matrix(R, d0=1.0):
    """
    Computes distance-dependent overlap parameters eta_nm(R).
    Returns 2x2 matrix:
        [[eta_11, eta_12],
         [eta_12, eta_22]]
    """
    eta_11 = (1.0 + R / d0 + (R**2) / (3.0 * d0**2)) * np.exp(-R / d0)
    eta_12 = 0.5 * (1.0 + R / d0 + (R**2) / (64.0 * d0**2)) * np.exp(-R / (2.0 * d0)) - 0.5 * np.exp(-R / d0)
    eta_22 = (1.0 + R / (2.0 * d0) + (R**2) / (24.0 * d0**2)) * np.exp(-R / (2.0 * d0))
    return np.array([[eta_11, eta_12],
                     [eta_12, eta_22]], dtype=float)

def calc_t_matrix(R, d0=1.0, t0=1.0):
    """
    Computes distance-dependent hopping parameters t_nm(R).
    Returns 2x2 matrix:
        [[t_11, t_12],
         [t_12, t_22]]
    """
    if np.isclose(R, 0.0):
        return np.zeros((2, 2), dtype=float)
    
    t_11 = -t0 * (1.0 + 2.0 * R / d0) * np.exp(-R / d0)
    t_12 = -t0 * (1.0 + 2.0 * R / d0) * np.exp(-R / d0) + t0 * (1.0 + R / (16.0 * d0)) * np.exp(-R / (2.0 * d0))
    t_22 = -t0 * (1.0 + R / (2.0 * d0) - (R**2) / (24.0 * d0**2)) * np.exp(-R / (2.0 * d0))
    return np.array([[t_11, t_12],
                     [t_12, t_22]], dtype=float)

# ==============================================================================
# BZ Paths
# ==============================================================================

def build_square_bz_path(num_seg=150):
    """
    Path: Gamma (0,0) -> X (pi, 0) -> M (pi, pi) -> Gamma (0,0) in dimensionless k*a units.
    """
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)
    path_ka = np.vstack([k1, k2, k3])
    path_x = np.linspace(0.0, 3.0, len(path_ka))
    tick_locs = [0.0, 1.0, 2.0, 3.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
    return path_ka, path_x, tick_locs, tick_labels

def build_rectangular_bz_path(a, b, num_seg=150):
    """
    Path: Gamma (0,0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0,0).
    """
    k_G = np.array([0.0, 0.0])
    k_X = np.array([np.pi / a, 0.0])
    k_M = np.array([np.pi / a, np.pi / b])
    k_Y = np.array([0.0, np.pi / b])

    k1 = np.linspace(k_G, k_X, num_seg, endpoint=False)
    k2 = np.linspace(k_X, k_M, num_seg, endpoint=False)
    k3 = np.linspace(k_M, k_Y, num_seg, endpoint=False)
    k4 = np.linspace(k_Y, k_G, num_seg + 1, endpoint=True)

    k_path = np.vstack([k1, k2, k3, k4])
    path_x = np.linspace(0.0, 4.0, len(k_path))
    tick_locs = [0.0, 1.0, 2.0, 3.0, 4.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r'$\Gamma$']
    return k_path, path_x, tick_locs, tick_labels

# ==============================================================================
# Square Lattice Dispersion Solver (User Formulation)
# ==============================================================================

def solve_square_lattice(a_val, path_ka, d0=1.0, t0=1.0, mode='Full'):
    """
    Computes E_1(k) and E_2(k) for square lattice using:
        M = eps_bar + t_bar @ inv(eta_bar)
        E = sort(real(eigvals(M)))

    Modes:
      'Full'       : NN + NNN with Overlap
      'No Overlap' : NN + NNN, eta_bar = Identity
      'NN only'    : NN with Overlap (omits NNN)
      'NNN only'   : NNN with Overlap (omits NN)
    """
    eps_bar = np.array([[0.0, 0.0],
                        [0.0, 0.8 * t0]], dtype=float)
    I2 = np.eye(2, dtype=float)

    inc_nn = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_nnn = (mode in ('Full', 'No Overlap', 'NNN only'))
    neglect_overlap = (mode == 'No Overlap')

    R_nn = a_val
    t_nn = calc_t_matrix(R_nn, d0=d0, t0=t0)
    eta_nn = calc_eta_matrix(R_nn, d0=d0)

    R_nnn = np.sqrt(2.0) * a_val
    t_nnn = calc_t_matrix(R_nnn, d0=d0, t0=t0)
    eta_nnn = calc_eta_matrix(R_nnn, d0=d0)

    num_pts = len(path_ka)
    energies = np.zeros((num_pts, 2), dtype=float)

    for i in range(num_pts):
        kx, ky = path_ka[i]
        eps_0 = np.cos(kx) + np.cos(ky)
        eps_1 = 2.0 * np.cos(kx) * np.cos(ky)

        t_k = np.zeros((2, 2), dtype=float)
        if inc_nn:
            t_k += 2.0 * t_nn * eps_0
        if inc_nnn:
            t_k += 2.0 * t_nnn * eps_1

        if neglect_overlap:
            M = eps_bar + t_k
        else:
            eta_k = I2.copy()
            if inc_nn:
                eta_k += 2.0 * eta_nn * eps_0
            if inc_nnn:
                eta_k += 2.0 * eta_nnn * eps_1
            M = eps_bar + t_k @ np.linalg.inv(eta_k)

        evals = np.linalg.eigvals(M)
        energies[i] = np.sort(np.real(evals))

    return energies

# ==============================================================================
# Rectangular Lattice Dispersion Solver (User Formulation)
# ==============================================================================

def solve_rectangular_lattice(a_val, b_val, k_path, d0=1.0, t0=1.0, mode='Full'):
    """
    Computes E_1(k), E_2(k), E_3(k) for Crystal B (3x3 basis) using:
        M = eps_bar + t_bar @ inv(eta_bar)
        E = sort(real(eigvals(M)))

    Modes:
      'Full'       : NN + NNN with Overlap (Intra NN + Intra NNN + Inter NN)
      'No Overlap' : Full hopping with eta_bar = Identity
      'NN only'    : Intra NN + Inter NN with Overlap (omits Intra NNN diagonal R_ab)
      'NNN only'   : Intra NNN with Overlap (omits NN couplings)
    """
    eps_bar = np.diag([0.0, 0.8 * t0, 0.0])
    I3 = np.eye(3, dtype=float)

    inc_nn = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_nnn = (mode in ('Full', 'No Overlap', 'NNN only'))
    neglect_overlap = (mode == 'No Overlap')

    R_a = a_val
    R_b = b_val
    R_ab = np.sqrt(a_val**2 + b_val**2)
    R_inter = 0.5 * R_ab

    eta_a = calc_eta_matrix(R_a, d0=d0)
    t_a = calc_t_matrix(R_a, d0=d0, t0=t0)

    eta_b = calc_eta_matrix(R_b, d0=d0)
    t_b = calc_t_matrix(R_b, d0=d0, t0=t0)

    eta_ab = calc_eta_matrix(R_ab, d0=d0)
    t_ab = calc_t_matrix(R_ab, d0=d0, t0=t0)

    eta_inter = calc_eta_matrix(R_inter, d0=d0)
    t_inter = calc_t_matrix(R_inter, d0=d0, t0=t0)

    num_pts = len(k_path)
    energies = np.zeros((num_pts, 3), dtype=float)

    for i in range(num_pts):
        kx, ky = k_path[i]

        cos_ax = np.cos(kx * a_val)
        cos_by = np.cos(ky * b_val)
        cos_diag = np.cos(kx * a_val) * np.cos(ky * b_val)
        cos_inter = 4.0 * np.cos(0.5 * kx * a_val) * np.cos(0.5 * ky * b_val)

        t_k = np.zeros((3, 3), dtype=float)
        eta_k = I3.copy() if not neglect_overlap else None

        # Intra-sublattice (1,1), (2,2), (1,2)
        term_t_11 = (2*t_a[0,0]*cos_ax + 2*t_b[0,0]*cos_by if inc_nn else 0.0) + (4*t_ab[0,0]*cos_diag if inc_nnn else 0.0)
        term_t_22 = (2*t_a[1,1]*cos_ax + 2*t_b[1,1]*cos_by if inc_nn else 0.0) + (4*t_ab[1,1]*cos_diag if inc_nnn else 0.0)
        term_t_12 = (2*t_a[0,1]*cos_ax + 2*t_b[0,1]*cos_by if inc_nn else 0.0) + (4*t_ab[0,1]*cos_diag if inc_nnn else 0.0)

        t_k[0, 0] = term_t_11
        t_k[1, 1] = term_t_22
        t_k[0, 1] = t_k[1, 0] = term_t_12
        t_k[2, 2] = term_t_11

        if not neglect_overlap:
            term_eta_11 = (2*eta_a[0,0]*cos_ax + 2*eta_b[0,0]*cos_by if inc_nn else 0.0) + (4*eta_ab[0,0]*cos_diag if inc_nnn else 0.0)
            term_eta_22 = (2*eta_a[1,1]*cos_ax + 2*eta_b[1,1]*cos_by if inc_nn else 0.0) + (4*eta_ab[1,1]*cos_diag if inc_nnn else 0.0)
            term_eta_12 = (2*eta_a[0,1]*cos_ax + 2*eta_b[0,1]*cos_by if inc_nn else 0.0) + (4*eta_ab[0,1]*cos_diag if inc_nnn else 0.0)

            eta_k[0, 0] += term_eta_11
            eta_k[1, 1] += term_eta_22
            eta_k[0, 1] = eta_k[1, 0] = term_eta_12
            eta_k[2, 2] = eta_k[0, 0]

        # Inter-sublattice (1,3), (2,3)
        if inc_nn:
            term_t_13 = t_inter[0, 0] * cos_inter
            term_t_23 = t_inter[0, 1] * cos_inter
            t_k[0, 2] = t_k[2, 0] = term_t_13
            t_k[1, 2] = t_k[2, 1] = term_t_23

            if not neglect_overlap:
                term_eta_13 = eta_inter[0, 0] * cos_inter
                term_eta_23 = eta_inter[0, 1] * cos_inter
                eta_k[0, 2] = eta_k[2, 0] = term_eta_13
                eta_k[1, 2] = eta_k[2, 1] = term_eta_23

        if neglect_overlap:
            M = eps_bar + t_k
        else:
            M = eps_bar + t_k @ np.linalg.inv(eta_k)

        evals = np.linalg.eigvals(M)
        energies[i] = np.sort(np.real(evals))

    return energies

# ==============================================================================
# Helper to Save to Both Locations
# ==============================================================================

TARGET_DIRS = [
    '/home/pseudonym/Documents/Code/Classes/Advanced_Solid_State_Stanescu_Fall2026/Advanced-Solid-State/HW2/plots',
    '/home/pseudonym/Documents/Code/Classes/Advanced_Solid_State_Stanescu_Fall2026/Advanced-Solid-State/HW2/Writeup/figures'
]

def save_fig_dual(fig, filename):
    for d in TARGET_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")

# ==============================================================================
# Figure Generation Functions
# ==============================================================================

MODES = ['Full', 'No Overlap', 'NN only', 'NNN only']
MODE_STYLES = {
    'Full':       {'color': '#1f77b4', 'ls': '-',  'lw': 2.2, 'label': 'Full (NN + NNN + Overlap)'},
    'No Overlap': {'color': '#d62728', 'ls': '--', 'lw': 2.0, 'label': 'Neglecting Overlap (S = I)'},
    'NN only':    {'color': '#2ca02c', 'ls': '-.', 'lw': 2.0, 'label': 'Neglecting NNN (NN only)'},
    'NNN only':   {'color': '#9467bd', 'ls': ':',  'lw': 2.2, 'label': 'Neglecting NN (NNN only)'}
}

def generate_square_figures():
    print("\n--- Generating Square Lattice Figures (User Inverse-Eta Formulation) ---")
    a_values = [5.0, 7.0, 9.0]
    path_ka, path_x, tick_locs, tick_labels = build_square_bz_path(num_seg=150)

    # 1. Individual 3-panel overview for each mode across a = 5, 7, 9
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=(mode != 'Full' and mode != 'NN only'))
        for idx, a_val in enumerate(a_values):
            ax = axes[idx]
            energies = solve_square_lattice(a_val, path_ka, mode=mode)
            ax.plot(path_x, energies[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
            ax.plot(path_x, energies[:, 1], 'r-', lw=2.2, label=r'$E_2(\vec{k})$ (Upper band)')

            ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = 0$')
            ax.axhline(0.8, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8\,t_0$')

            for t in tick_locs:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_title(rf'$a = {a_val:.1f}\,d_0$', fontsize=13, fontweight='bold')
            ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
            if idx == 0:
                ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

            if a_val == 5.0 and mode in ('Full', 'NN only', 'NNN only'):
                ax.set_ylim(-40, 270)
            elif a_val >= 7.0:
                ax.set_ylim(-0.3, 1.6)

        mode_slug = mode.lower().replace(' ', '_').replace('+', 'and')
        fig.suptitle(f'Square Lattice Band Structure: {mode} Model (User Formulation: $M = \\bar{{\\epsilon}} + \\bar{{t}}\\bar{{\\eta}}^{{-1}}$)',
                     fontsize=14, fontweight='bold', y=1.02)
        save_fig_dual(fig, f'square_lattice_{mode_slug}_bands.png')
        if mode == 'Full':
            save_fig_dual(fig, 'band_structures_part1.png')
        plt.close(fig)

    # 2. Detailed comparison figures for each lattice constant a = 5, 7, 9
    for a_val in a_values:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for mode in MODES:
            energies = solve_square_lattice(a_val, path_ka, mode=mode)
            st = MODE_STYLES[mode]
            ax.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
            ax.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])

        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
        for t in tick_locs:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=13)
        ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=12)
        ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
        ax.set_title(rf'Square Lattice Approximations Comparison at $a = {a_val:.1f}\,d_0$', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

        # Set zoom limits to see bands clearly while handling divergence at a=5.0
        if a_val == 5.0:
            ax.set_ylim(-1.5, 2.5)
            ax.text(0.03, 0.93, 'Note: Overlap models diverge off-scale\nalong M-Gamma line due to singular eta(k)',
                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif a_val == 7.0:
            ax.set_ylim(-0.3, 1.5)
        else:
            ax.set_ylim(-0.1, 1.0)

        save_fig_dual(fig, f'square_comparison_a{int(a_val)}.png')
        if a_val == 5.0:
            save_fig_dual(fig, 'band_structure_comparisons.png')
        plt.close(fig)

    # 3. Comprehensive 3-panel comparison figure across all a values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, a_val in enumerate(a_values):
        ax = axes[idx]
        for mode in MODES:
            energies = solve_square_lattice(a_val, path_ka, mode=mode)
            st = MODE_STYLES[mode]
            ax.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
            ax.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])

        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
        for t in tick_locs:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=12)
        ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
        if idx == 0:
            ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
        ax.set_title(rf'$a = {a_val:.1f}\,d_0$', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)

        if a_val == 5.0:
            ax.set_ylim(-1.5, 2.5)
        elif a_val == 7.0:
            ax.set_ylim(-0.3, 1.5)
        else:
            ax.set_ylim(-0.1, 1.0)

        if idx == 1:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    fig.suptitle('Square Lattice: Comparison of Physical Approximations across Lattice Parameters\n(User Formulation: $M = \\bar{\\epsilon} + \\bar{t}\\bar{\\eta}^{-1}$)',
                 fontsize=14, fontweight='bold', y=1.03)
    save_fig_dual(fig, 'square_lattice_comparisons_all_a.png')
    plt.close(fig)

def generate_rectangular_figures():
    print("\n--- Generating Rectangular Lattice Figures (Crystal B - User Inverse-Eta Formulation) ---")
    a_values = [5.0, 7.0, 9.0]
    b_ratio = 1.2

    # 1. Individual 3-panel overview for each mode across a = 5, 7, 9
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
        for idx, a_val in enumerate(a_values):
            b_val = b_ratio * a_val
            ax = axes[idx]
            k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
            energies = solve_rectangular_lattice(a_val, b_val, k_path, mode=mode)

            ax.plot(path_x, energies[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
            ax.plot(path_x, energies[:, 1], 'g-', lw=2.2, label=r'$E_2(\vec{k})$ (Middle band)')
            ax.plot(path_x, energies[:, 2], 'r-', lw=2.2, label=r'$E_3(\vec{k})$ (Upper band)')

            ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = \epsilon_3 = 0$')
            ax.axhline(0.8, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8\,t_0$')

            for t in tick_locs:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_title(rf'$a = {a_val:.1f}\,d_0$, $b = {b_val:.1f}\,d_0$', fontsize=12, fontweight='bold')
            ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
            if idx == 0:
                ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=8.5)

            if a_val == 5.0 and mode in ('Full', 'NN only'):
                ax.set_ylim(-35, 45)
            elif a_val == 7.0:
                ax.set_ylim(-0.8, 1.8)
            else:
                ax.set_ylim(-0.3, 1.0)

        mode_slug = mode.lower().replace(' ', '_').replace('+', 'and')
        fig.suptitle(f'Crystal B (Centered Rectangular) Band Structure: {mode} Model\n(User Formulation: $M = \\bar{{\\epsilon}} + \\bar{{t}}\\bar{{\\eta}}^{{-1}}$)',
                     fontsize=14, fontweight='bold', y=1.04)
        save_fig_dual(fig, f'rectangular_lattice_{mode_slug}_bands.png')
        if mode == 'Full':
            save_fig_dual(fig, 'band_structures_lattice_b.png')
        plt.close(fig)

    # 2. Detailed comparison figures for each lattice constant a = 5, 7, 9
    for a_val in a_values:
        b_val = b_ratio * a_val
        k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
        fig, ax = plt.subplots(figsize=(11, 6.8))

        for mode in MODES:
            energies = solve_rectangular_lattice(a_val, b_val, k_path, mode=mode)
            st = MODE_STYLES[mode]
            ax.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
            ax.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])
            ax.plot(path_x, energies[:, 2], color=st['color'], ls=st['ls'], lw=st['lw'])

        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
        for t in tick_locs:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=13)
        ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=12)
        ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
        ax.set_title(rf'Crystal B Approximations Comparison at $a = {a_val:.1f}\,d_0$, $b = {b_val:.1f}\,d_0$', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95)

        if a_val == 5.0:
            ax.set_ylim(-2.0, 3.0)
            ax.text(0.03, 0.93, 'Note: Overlap models exhibit singularities off-scale at a=5.0 d0',
                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        elif a_val == 7.0:
            ax.set_ylim(-0.8, 1.8)
        else:
            ax.set_ylim(-0.3, 1.0)

        save_fig_dual(fig, f'rectangular_comparison_a{int(a_val)}.png')
        if a_val == 7.0:
            save_fig_dual(fig, 'band_structure_comparisons_lattice_b.png')
        plt.close(fig)

    # 3. Comprehensive 3-panel comparison figure across all a values
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for idx, a_val in enumerate(a_values):
        b_val = b_ratio * a_val
        k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
        ax = axes[idx]

        for mode in MODES:
            energies = solve_rectangular_lattice(a_val, b_val, k_path, mode=mode)
            st = MODE_STYLES[mode]
            ax.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
            ax.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])
            ax.plot(path_x, energies[:, 2], color=st['color'], ls=st['ls'], lw=st['lw'])

        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
        for t in tick_locs:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=12)
        ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
        if idx == 0:
            ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
        ax.set_title(rf'$a = {a_val:.1f}\,d_0$, $b = {b_val:.1f}\,d_0$', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        if a_val == 5.0:
            ax.set_ylim(-2.0, 3.0)
        elif a_val == 7.0:
            ax.set_ylim(-0.8, 1.8)
        else:
            ax.set_ylim(-0.3, 1.0)

        if idx == 1:
            ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    fig.suptitle('Crystal B (Centered Rectangular): Comparison of Physical Approximations\n(User Formulation: $M = \\bar{\\epsilon} + \\bar{t}\\bar{\\eta}^{-1}$)',
                 fontsize=14, fontweight='bold', y=1.03)
    save_fig_dual(fig, 'rectangular_lattice_comparisons_all_a.png')
    plt.close(fig)

def main():
    print("=" * 70)
    print("HW2 Master Figure Generator: Executing User Inverse-Eta Formulation")
    print("=" * 70)
    generate_square_figures()
    generate_rectangular_figures()
    print("\n[SUCCESS] All figures generated and saved to:")
    for d in TARGET_DIRS:
        print(f"  - {d}")
    print("=" * 70)

if __name__ == '__main__':
    main()
