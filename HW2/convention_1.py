#!/usr/bin/env python3
"""
================================================================================
convention_1.py: Tight-Binding Model under Convention 1 (Global Distance Shells)
================================================================================

Description:
  Implements the tight-binding model for Crystal A (Square Lattice) and
  Crystal B (Centered Rectangular Lattice) where neighbor shells are strictly
  classified by GLOBAL real-space distance R from any given atom.

Physical Geometry for Crystal B (b = 1.5 * a):
  Basis vectors: a1 = (a, 0), a2 = (0, b) with b = 1.5 * a.
  Site 1: tau_1 = (0, 0), with Orbitals 1 and 2 (Basis states 1 and 2).
  Site 2: tau_2 = (a/2, b/2), with Orbital 1 only (Basis state 3).

Neighbor Shells by Ascending Real-Space Distance R from Site 1:
  - On-site (R = 0):
      Site 1 self-interactions (eta_11 = eta_22 = 1, eta_12 = 0, t_nm = 0).
      Site 2 self-interactions (eta_33 = 1, t_33 = 0).
      No inter-sublattice bonds exist at R = 0.

  - Global Shell 1 (Nearest Neighbors - NN):
      Distance r_1 = 0.5 * sqrt(a^2 + b^2) approx 0.90139 * a.
      Contains 4 Site 2 atoms at (+-a/2, +-b/2) around Site 1.
      Bravais vectors: R_j in {(0,0), (-a,0), (0,-b), (-a,-b)}.
      Couples: Inter-sublattice blocks (1,3) and (2,3).
      Phase factor: (1 + e^{-i*kx*a}) * (1 + e^{-i*ky*b}).
      *Note*: 0 intra-sublattice bonds exist at this distance.

  - Global Shell 2 (Next-Nearest Neighbors - NNN):
      Distance r_2 = a.
      Contains 2 atoms at (+-a, 0) along the x-axis.
      Couples: Intra-sublattice blocks (1,1), (2,2), (1,2) on Site 1,
               and (3,3) on Site 2.
      Phase factor: 2 * cos(kx * a).
      *Note*: Distance b = 1.5*a along y is r_3 > r_2, and is therefore
      EXCLUDED under the strict 2-shell global cutoff.

Approximation Modes:
  - 'Full': Global Shell 1 (r_1) + Global Shell 2 (r_2 = a) with Overlap.
  - 'No Overlap': Global Shell 1 + Shell 2 with S = I.
  - 'NN only': Global Shell 1 only (purely inter-sublattice; intra hopping is zero).

Outputs:
  Plots are saved to 'plots/convention_1/' and 'Writeup/figures/convention_1/'.
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# Model Parameters and Units
# ==============================================================================

D0 = 1.0  # Characteristic length unit d_0
T0 = 1.0  # Characteristic energy unit t_0

MODES = ['Full', 'No Overlap', 'NN only']
MODE_STYLES = {
    'Full':       {'color': '#1f77b4', 'ls': '-',  'lw': 2.2, 'label': 'Full (Global Shell 1 + 2 + Overlap)'},
    'No Overlap': {'color': '#d62728', 'ls': '--', 'lw': 2.0, 'label': 'No Overlap (S = I)'},
    'NN only':    {'color': '#2ca02c', 'ls': '-.', 'lw': 2.0, 'label': 'NN only (Global Shell 1 only)'}
}

TARGET_DIRS = [
    os.path.join(os.path.dirname(__file__), 'plots', 'convention_1'),
    os.path.join(os.path.dirname(__file__), 'Writeup', 'figures', 'convention_1')
]

# ==============================================================================
# Radial Overlap and Hopping Matrix Functions
# ==============================================================================

def calc_eta_matrix(R, d0=D0):
    """Evaluates 2x2 overlap matrix eta_nm(R) between orbitals 1 and 2."""
    R = np.maximum(np.asarray(R, dtype=float), 1e-15)
    r_d0 = R / d0
    r_2d0 = R / (2.0 * d0)

    eta_11 = (1.0 + r_d0 + (R**2) / (3.0 * d0**2)) * np.exp(-r_d0)
    eta_12 = 0.5 * (1.0 + r_d0 + (R**2) / (64.0 * d0**2)) * np.exp(-r_2d0) - 0.5 * np.exp(-r_d0)
    eta_22 = (1.0 + r_2d0 + (R**2) / (24.0 * d0**2)) * np.exp(-r_2d0)
    return np.array([[eta_11, eta_12], [eta_12, eta_22]], dtype=float)

def calc_t_matrix(R, d0=D0, t0=T0):
    """Evaluates 2x2 transfer matrix t_nm(R) between orbitals 1 and 2."""
    R = np.maximum(np.asarray(R, dtype=float), 1e-15)
    r_d0 = R / d0
    r_2d0 = R / (2.0 * d0)

    t_11 = -t0 * (1.0 + 2.0 * r_d0) * np.exp(-r_d0)
    t_12 = -t0 * (1.0 + 2.0 * r_d0) * np.exp(-r_d0) + t0 * (1.0 + R / (16.0 * d0)) * np.exp(-r_2d0)
    t_22 = -t0 * (1.0 + r_2d0 - (R**2) / (24.0 * d0**2)) * np.exp(-r_2d0)
    return np.array([[t_11, t_12], [t_12, t_22]], dtype=float)

# ==============================================================================
# Brillouin Zone Paths
# ==============================================================================

def build_square_bz_path(num_seg=150):
    """High-symmetry path Gamma -> X -> M -> Gamma for Crystal A."""
    corners = np.array([
        [0.0, 0.0],
        [np.pi, 0.0],
        [np.pi, np.pi],
        [0.0, 0.0]
    ])
    labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']

    k_list = []
    path_dists = [0.0]
    tick_locs = [0.0]

    for s in range(len(corners) - 1):
        p_start = corners[s]
        p_end = corners[s + 1]
        seg_len = np.linalg.norm(p_end - p_start)
        t_vals = np.linspace(0.0, 1.0, num_seg, endpoint=(s == len(corners) - 2))

        for t in t_vals:
            k_pt = (1.0 - t) * p_start + t * p_end
            if len(k_list) > 0:
                step_len = np.linalg.norm(k_pt - k_list[-1])
                path_dists.append(path_dists[-1] + step_len)
            k_list.append(k_pt)
        tick_locs.append(path_dists[-1])

    return np.array(k_list), np.array(path_dists), tick_locs, labels

def build_rectangular_bz_path(a, b, num_seg=150):
    """High-symmetry path Gamma -> X -> M -> Y -> Gamma for Crystal B."""
    corners = np.array([
        [0.0, 0.0],
        [np.pi / a, 0.0],
        [np.pi / a, np.pi / b],
        [0.0, np.pi / b],
        [0.0, 0.0]
    ])
    labels = [r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r'$\Gamma$']

    k_list = []
    path_dists = [0.0]
    tick_locs = [0.0]

    for s in range(len(corners) - 1):
        p_start = corners[s]
        p_end = corners[s + 1]
        seg_len = np.linalg.norm(p_end - p_start)
        t_vals = np.linspace(0.0, 1.0, num_seg, endpoint=(s == len(corners) - 2))

        for t in t_vals:
            k_pt = (1.0 - t) * p_start + t * p_end
            if len(k_list) > 0:
                step_len = np.linalg.norm(k_pt - k_list[-1])
                path_dists.append(path_dists[-1] + step_len)
            k_list.append(k_pt)
        tick_locs.append(path_dists[-1])

    return np.array(k_list), np.array(path_dists), tick_locs, labels

# ==============================================================================
# Solvers: Crystal A (Square Lattice)
# ==============================================================================

def solve_square_point(kx_a, ky_a, a_val, t_nn, eta_nn, t_nnn, eta_nnn, eps_bar, I2, mode='Full'):
    inc_nn = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_nnn = (mode in ('Full', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    eps_0 = np.cos(kx_a) + np.cos(ky_a)
    eps_1 = 2.0 * np.cos(kx_a) * np.cos(ky_a)

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
    return np.sort(np.real(evals))

def solve_square_path(a_val, path_ka, mode='Full'):
    eps_bar = np.array([[0.0, 0.0], [0.0, 0.8 * T0]], dtype=float)
    I2 = np.eye(2, dtype=float)
    t_nn = calc_t_matrix(a_val, d0=D0, t0=T0)
    eta_nn = calc_eta_matrix(a_val, d0=D0)
    t_nnn = calc_t_matrix(np.sqrt(2.0) * a_val, d0=D0, t0=T0)
    eta_nnn = calc_eta_matrix(np.sqrt(2.0) * a_val, d0=D0)

    num_pts = len(path_ka)
    energies = np.zeros((num_pts, 2), dtype=float)
    for i in range(num_pts):
        energies[i] = solve_square_point(path_ka[i, 0], path_ka[i, 1], a_val,
                                         t_nn, eta_nn, t_nnn, eta_nnn, eps_bar, I2, mode=mode)
    return energies

# ==============================================================================
# Solvers: Crystal B under Convention 1 (Global Distance Shells)
# ==============================================================================

def get_convention1_params(a_val, b_val, d0=D0, t0=T0):
    """
    Precomputes matrices for Convention 1 shells:
      Shell 1 (NN):  r_1 = 0.5 * sqrt(a^2 + b^2) [Inter-sublattice 1<->2]
      Shell 2 (NNN): r_2 = a                     [Intra-sublattice along x]
    """
    r_1 = 0.5 * np.sqrt(a_val**2 + b_val**2)
    r_2 = a_val

    return {
        'r_1': r_1,
        'r_2': r_2,
        't_shell1': calc_t_matrix(r_1, d0=d0, t0=t0),
        'eta_shell1': calc_eta_matrix(r_1, d0=d0),
        't_shell2': calc_t_matrix(r_2, d0=d0, t0=t0),
        'eta_shell2': calc_eta_matrix(r_2, d0=d0),
        'eps_bar': np.diag([0.0, 0.8 * t0, 0.0]),
        'I3': np.eye(3, dtype=complex)
    }

def solve_rectangular_point_conv1(kx, ky, a_val, b_val, p, mode='Full'):
    """
    Evaluates 3x3 M = eps_bar + t_bar @ inv(eta_bar) under Convention 1:
      - Shell 1 (NN): Inter-sublattice 4 center atoms at r_1 = 0.5*sqrt(a^2+b^2)
      - Shell 2 (NNN): Intra-sublattice 2 atoms at r_2 = a (x-axis ONLY)
      - Notice: 0 hopping along y (b = 1.5*a > a is excluded as Shell 3).
    """
    inc_shell1 = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_shell2 = (mode in ('Full', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    t_k = np.zeros((3, 3), dtype=complex)
    eta_k = p['I3'].copy() if not neglect_overlap else None

    # Intra-sublattice: Shell 2 at distance r_2 = a along x ONLY
    if inc_shell2:
        cos_ax = np.cos(kx * a_val)
        term_t_11 = 2.0 * p['t_shell2'][0, 0] * cos_ax
        term_t_22 = 2.0 * p['t_shell2'][1, 1] * cos_ax
        term_t_12 = 2.0 * p['t_shell2'][0, 1] * cos_ax

        t_k[0, 0] = term_t_11
        t_k[1, 1] = term_t_22
        t_k[0, 1] = term_t_12
        t_k[1, 0] = term_t_12
        t_k[2, 2] = term_t_11  # Site 2 has identical x-neighbors

        if not neglect_overlap:
            term_eta_11 = 2.0 * p['eta_shell2'][0, 0] * cos_ax
            term_eta_22 = 2.0 * p['eta_shell2'][1, 1] * cos_ax
            term_eta_12 = 2.0 * p['eta_shell2'][0, 1] * cos_ax

            eta_k[0, 0] += term_eta_11
            eta_k[1, 1] += term_eta_22
            eta_k[0, 1] += term_eta_12
            eta_k[1, 0] += term_eta_12
            eta_k[2, 2] += term_eta_11

    # Inter-sublattice: Shell 1 at distance r_1 = 0.5*sqrt(a^2+b^2)
    if inc_shell1:
        ph_shell1 = (1.0 + np.exp(-1j * a_val * kx)) * (1.0 + np.exp(-1j * b_val * ky))
        t_13 = p['t_shell1'][0, 0] * ph_shell1
        t_23 = p['t_shell1'][0, 1] * ph_shell1

        t_k[0, 2] = t_13
        t_k[2, 0] = np.conj(t_13)
        t_k[1, 2] = t_23
        t_k[2, 1] = np.conj(t_23)

        if not neglect_overlap:
            eta_13 = p['eta_shell1'][0, 0] * ph_shell1
            eta_23 = p['eta_shell1'][0, 1] * ph_shell1

            eta_k[0, 2] = eta_13
            eta_k[2, 0] = np.conj(eta_13)
            eta_k[1, 2] = eta_23
            eta_k[2, 1] = np.conj(eta_23)

    if neglect_overlap:
        M = p['eps_bar'] + t_k
    else:
        M = p['eps_bar'] + t_k @ np.linalg.inv(eta_k)

    evals = np.linalg.eigvals(M)
    return np.sort(np.real(evals))

def solve_rectangular_path_conv1(a_val, b_val, k_path, mode='Full'):
    p = get_convention1_params(a_val, b_val)
    num_pts = len(k_path)
    energies = np.zeros((num_pts, 3), dtype=float)
    for i in range(num_pts):
        energies[i] = solve_rectangular_point_conv1(k_path[i, 0], k_path[i, 1], a_val, b_val, p, mode=mode)
    return energies

# ==============================================================================
# Batch Figure Reproduction
# ==============================================================================

def save_fig_dual(fig, filename):
    for d in TARGET_DIRS:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")

def reproduce_all_figures():
    """Reproduces all figures under Convention 1 (Global Distance Shells)."""
    print("=" * 72)
    print("Executing Batch Reproduction: CONVENTION 1 (Global Distance Shells)")
    print(f"Parameters: Fixed t_0 = {T0}, d_0 = {D0}, b = 1.5 * a")
    print("=" * 72)

    a_values = [5.0, 7.0, 9.0]
    b_ratio = 1.5

    # 1. Square Lattice Figures (Convention 1 == Convention 2 for 1-site basis)
    print("\n[1/2] Processing Square Lattice (Crystal A)...")
    path_ka, path_x, tick_locs, tick_labels = build_square_bz_path(num_seg=150)

    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
        for idx, a_val in enumerate(a_values):
            ax = axes[idx]
            energies = solve_square_path(a_val, path_ka, mode=mode)
            ax.plot(path_x, energies[:, 0], 'b-', lw=2.2)
            ax.plot(path_x, energies[:, 1], 'r-', lw=2.2)
            ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4)
            ax.axhline(0.8, color='red', linestyle=':', alpha=0.4)
            for t in tick_locs:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_title(rf'$a = {a_val:.1f}\,d_0$', fontsize=12, fontweight='bold')
            ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
            if idx == 0:
                ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
            ax.grid(alpha=0.3)
            if mode in ('Full', 'NN only') and a_val == 5.0:
                ax.set_ylim(-15, 25)
            else:
                ax.set_ylim(-0.8, 1.8)

        mode_slug = mode.lower().replace(' ', '_')
        fig.suptitle(f'Crystal A: {mode} (Convention 1)', fontsize=14, fontweight='bold', y=1.02)
        save_fig_dual(fig, f'square_lattice_{mode_slug}_bands.png')
        plt.close(fig)

    # 2. Rectangular Lattice Figures (Convention 1)
    print("\n[2/2] Processing Rectangular Lattice (Crystal B, Convention 1)...")
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
        for idx, a_val in enumerate(a_values):
            b_val = b_ratio * a_val
            ax = axes[idx]
            k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
            energies = solve_rectangular_path_conv1(a_val, b_val, k_path, mode=mode)

            ax.plot(path_x, energies[:, 0], 'b-', lw=2.2)
            ax.plot(path_x, energies[:, 1], 'g-', lw=2.2)
            ax.plot(path_x, energies[:, 2], 'r-', lw=2.2)
            ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4)
            ax.axhline(0.8, color='red', linestyle=':', alpha=0.4)
            for t in tick_locs:
                ax.axvline(t, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels, fontsize=12)
            ax.set_title(rf'$a = {a_val:.1f}\,d_0$, $b = {b_val:.1f}\,d_0$', fontsize=12, fontweight='bold')
            ax.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
            if idx == 0:
                ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=12)
            ax.grid(alpha=0.3)

            if mode in ('Full', 'NN only') and a_val == 5.0:
                ax.set_ylim(-15, 25)
            else:
                ax.set_ylim(-0.8, 1.8)

        mode_slug = mode.lower().replace(' ', '_')
        fig.suptitle(f'Crystal B: {mode} (Convention 1: Global Distance Shells)', fontsize=14, fontweight='bold', y=1.02)
        save_fig_dual(fig, f'rectangular_lattice_{mode_slug}_bands.png')
        plt.close(fig)

    # Comparison across approximations at a = 7.0
    a_val = 7.0
    b_val = b_ratio * a_val
    k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for mode in MODES:
        energies = solve_rectangular_path_conv1(a_val, b_val, k_path, mode=mode)
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
    ax.set_title(rf'Crystal B Approximations (Convention 1) at $a = {a_val:.1f}\,d_0$, $b = {b_val:.1f}\,d_0$', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.8, 1.8)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    save_fig_dual(fig, 'band_structure_comparisons_lattice_b.png')
    plt.close(fig)

    print("\n[SUCCESS] Convention 1 batch generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tight-Binding Solvers under Convention 1 (Global Shells)")
    parser.add_argument('--batch', action='store_true', help="Generate and save publication figures")
    args = parser.parse_args()

    if args.batch:
        reproduce_all_figures()
    else:
        print("Run with --batch to generate all figures for Convention 1.")
