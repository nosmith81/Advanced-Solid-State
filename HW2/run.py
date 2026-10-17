#!/usr/bin/env python3
"""
run.py: Consolidated electronic band structure calculation and interactive visualization
for Square Lattice (Crystal A) and Centered Rectangular Lattice (Crystal B).

Key Specifications:
- Formulated strictly using the user's standardized eigenvalue problem:
    M(k) = eps_bar + t_bar(k) @ inv(eta_bar(k))
    M c' = E c'
    evals = np.linalg.eigvals(M)
    E(k) = np.sort(np.real(evals))
- Energy parameter fixed to t_0 = 1.0 (no slider).
- Interactive mode (via --interactive or -i):
    * Lattice parameter slider: a / d_0
    * Two sets of RadioButtons:
        1. Lattice selection: ('Square', 'Rectangular')
        2. View mode: ('High-Symmetry Path', '3D Brillouin Zone')
    * CheckButtons for approximations:
        - Full (NN + NNN + Overlap)
        - No Overlap (S = I)
        - NN only (Neglect NNN)
        - NNN only (Neglect NN)
- Batch mode (default or via --batch):
    * Reproduces all figures across all approximations and lattice parameters
      (a = 5.0, 7.0, 9.0 d_0) for both lattices.
    * Saves outputs to both HW2/plots/ and HW2/Writeup/figures/.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

# ==============================================================================
# Model Parameters and Spatial Functions
# ==============================================================================

T0 = 1.0  # Fixed energy scale
D0 = 1.0  # Fixed length scale

def calc_eta_matrix(R, d0=D0):
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

def calc_t_matrix(R, d0=D0, t0=T0):
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
# Brillouin Zone Paths and Meshes
# ==============================================================================

def build_square_bz_path(num_seg=150):
    """Path: Gamma (0,0) -> X (pi, 0) -> M (pi, pi) -> Gamma (0,0) (dimensionless k*a)."""
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)
    path_ka = np.vstack([k1, k2, k3])
    path_x = np.linspace(0.0, 3.0, len(path_ka))
    tick_locs = [0.0, 1.0, 2.0, 3.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']
    return path_ka, path_x, tick_locs, tick_labels

def build_rectangular_bz_path(a, b, num_seg=150):
    """Path: Gamma (0,0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0,0)."""
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

def build_square_bz_mesh(a, grid_pts=35):
    """Generates 2D mesh (KX, KY) over [-pi/a, pi/a] x [-pi/a, pi/a]."""
    k = np.linspace(-np.pi / a, np.pi / a, grid_pts)
    return np.meshgrid(k, k)

def build_rectangular_bz_mesh(a, b, grid_pts=35):
    """Generates 2D mesh (KX, KY) over [-pi/a, pi/a] x [-pi/b, pi/b]."""
    kx = np.linspace(-np.pi / a, np.pi / a, grid_pts)
    ky = np.linspace(-np.pi / b, np.pi / b, grid_pts)
    return np.meshgrid(kx, ky)

# ==============================================================================
# Solvers: Square Lattice (Crystal A - 2x2)
# ==============================================================================

def solve_square_point(kx_a, ky_a, a_val, t_nn, eta_nn, t_nnn, eta_nnn, eps_bar, I2, mode='Full'):
    """Evaluates 2x2 M = eps_bar + t_bar @ inv(eta_bar) at a single (kx*a, ky*a)."""
    inc_nn = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_nnn = (mode in ('Full', 'No Overlap', 'NNN only'))
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

solve_square_lattice = solve_square_path

def solve_square_mesh(a_val, KX, KY, mode='Full'):
    eps_bar = np.array([[0.0, 0.0], [0.0, 0.8 * T0]], dtype=float)
    I2 = np.eye(2, dtype=float)
    t_nn = calc_t_matrix(a_val, d0=D0, t0=T0)
    eta_nn = calc_eta_matrix(a_val, d0=D0)
    t_nnn = calc_t_matrix(np.sqrt(2.0) * a_val, d0=D0, t0=T0)
    eta_nnn = calc_eta_matrix(np.sqrt(2.0) * a_val, d0=D0)

    E1 = np.zeros_like(KX)
    E2 = np.zeros_like(KX)
    rows, cols = KX.shape
    for r in range(rows):
        for c in range(cols):
            ev = solve_square_point(KX[r, c] * a_val, KY[r, c] * a_val, a_val,
                                    t_nn, eta_nn, t_nnn, eta_nnn, eps_bar, I2, mode=mode)
            E1[r, c] = ev[0]
            E2[r, c] = ev[1]
    return E1, E2

# ==============================================================================
# Solvers: Rectangular Lattice (Crystal B - 3x3)
# ==============================================================================

def solve_rectangular_point(kx, ky, a_val, b_val, params_b, mode='Full'):
    """Evaluates 3x3 M = eps_bar + t_bar @ inv(eta_bar) at a single (kx, ky)."""
    inc_nn = (mode in ('Full', 'No Overlap', 'NN only'))
    inc_nnn = (mode in ('Full', 'No Overlap', 'NNN only'))
    neglect_overlap = (mode == 'No Overlap')

    t_a, eta_a, t_b, eta_b, t_ab, eta_ab, t_inter, eta_inter, eps_bar, I3 = params_b

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
        t_k[0, 2] = t_k[2, 0] = t_inter[0, 0] * cos_inter
        t_k[1, 2] = t_k[2, 1] = t_inter[0, 1] * cos_inter
        if not neglect_overlap:
            eta_k[0, 2] = eta_k[2, 0] = eta_inter[0, 0] * cos_inter
            eta_k[1, 2] = eta_k[2, 1] = eta_inter[0, 1] * cos_inter

    if neglect_overlap:
        M = eps_bar + t_k
    else:
        M = eps_bar + t_k @ np.linalg.inv(eta_k)

    evals = np.linalg.eigvals(M)
    return np.sort(np.real(evals))

def get_rectangular_params(a_val, b_val):
    R_a = a_val
    R_b = b_val
    R_ab = np.sqrt(a_val**2 + b_val**2)
    R_inter = 0.5 * R_ab
    return (
        calc_t_matrix(R_a, d0=D0, t0=T0),
        calc_eta_matrix(R_a, d0=D0),
        calc_t_matrix(R_b, d0=D0, t0=T0),
        calc_eta_matrix(R_b, d0=D0),
        calc_t_matrix(R_ab, d0=D0, t0=T0),
        calc_eta_matrix(R_ab, d0=D0),
        calc_t_matrix(R_inter, d0=D0, t0=T0),
        calc_eta_matrix(R_inter, d0=D0),
        np.diag([0.0, 0.8 * T0, 0.0]),
        np.eye(3, dtype=float)
    )

def solve_rectangular_path(a_val, b_val, k_path, mode='Full'):
    params_b = get_rectangular_params(a_val, b_val)
    num_pts = len(k_path)
    energies = np.zeros((num_pts, 3), dtype=float)
    for i in range(num_pts):
        energies[i] = solve_rectangular_point(k_path[i, 0], k_path[i, 1], a_val, b_val, params_b, mode=mode)
    return energies

solve_rectangular_lattice = solve_rectangular_path

def solve_rectangular_mesh(a_val, b_val, KX, KY, mode='Full'):
    params_b = get_rectangular_params(a_val, b_val)
    E1 = np.zeros_like(KX)
    E2 = np.zeros_like(KX)
    E3 = np.zeros_like(KX)
    rows, cols = KX.shape
    for r in range(rows):
        for c in range(cols):
            ev = solve_rectangular_point(KX[r, c], KY[r, c], a_val, b_val, params_b, mode=mode)
            E1[r, c] = ev[0]
            E2[r, c] = ev[1]
            E3[r, c] = ev[2]
    return E1, E2, E3

# ==============================================================================
# Interactive GUI Application
# ==============================================================================

MODES = ['Full', 'No Overlap', 'NN only', 'NNN only']
CHECK_LABELS = [
    'Full (NN + NNN + Overlap)',
    'No Overlap (S = I)',
    'NN only (Neglect NNN)',
    'NNN only (Neglect NN)'
]
MODE_MAP = dict(zip(CHECK_LABELS, MODES))

MODE_STYLES = {
    'Full':       {'color': '#1f77b4', 'ls': '-',  'lw': 2.2, 'label': 'Full (NN + NNN + Overlap)'},
    'No Overlap': {'color': '#d62728', 'ls': '--', 'lw': 2.0, 'label': 'Neglecting Overlap (S = I)'},
    'NN only':    {'color': '#2ca02c', 'ls': '-.', 'lw': 2.0, 'label': 'Neglecting NNN (NN only)'},
    'NNN only':   {'color': '#9467bd', 'ls': ':',  'lw': 2.2, 'label': 'Neglecting NN (NNN only)'}
}

SURF_CMAPS = {
    'Full': 'Blues_r',
    'No Overlap': 'Reds_r',
    'NN only': 'Greens_r',
    'NNN only': 'Purples_r'
}

def run_interactive():
    print("Launching Interactive Solid State Tight-Binding Explorer...")
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title("HW2 Tight-Binding Band Structure Explorer")

    # Dual axes in the main viewport: 2D path vs 3D BZ surface
    ax_2d = fig.add_axes([0.08, 0.16, 0.62, 0.77])
    ax_3d = fig.add_axes([0.08, 0.16, 0.62, 0.77], projection='3d')
    ax_3d.set_visible(False)

    # Controls area (Right & Bottom)
    # 1. Slider: Lattice parameter a
    ax_a = fig.add_axes([0.12, 0.05, 0.54, 0.035])
    a_slider = Slider(ax_a, r'Lattice $a / d_0$', 3.5, 12.0, valinit=7.0, valstep=0.1)

    # 2. Radio Button Set 1: Lattice geometry
    ax_lat = fig.add_axes([0.74, 0.75, 0.23, 0.16])
    radio_lattice = RadioButtons(ax_lat, ('Square', 'Rectangular'), active=0)
    ax_lat.set_title("Lattice Geometry", fontsize=11, fontweight='bold', pad=8)

    # 3. Radio Button Set 2: View mode
    ax_view = fig.add_axes([0.74, 0.52, 0.23, 0.16])
    radio_view = RadioButtons(ax_view, ('High-Symmetry Path', '3D Brillouin Zone'), active=0)
    ax_view.set_title("View Mode", fontsize=11, fontweight='bold', pad=8)

    # 4. CheckBoxes: Physical Approximations
    ax_chk = fig.add_axes([0.74, 0.16, 0.24, 0.28])
    chk_approx = CheckButtons(ax_chk, CHECK_LABELS, [True, True, False, False])
    ax_chk.set_title("Approximations Included", fontsize=11, fontweight='bold', pad=8)

    def update(val=None):
        a_val = a_slider.val
        lat_choice = radio_lattice.value_selected
        view_choice = radio_view.value_selected
        b_val = 1.2 * a_val

        # Determine checked modes
        statuses = chk_approx.get_status()
        active_modes = [MODE_MAP[lbl] for lbl, st in zip(CHECK_LABELS, statuses) if st]

        if view_choice == 'High-Symmetry Path':
            ax_3d.set_visible(False)
            ax_2d.set_visible(True)
            ax_2d.clear()

            if not active_modes:
                ax_2d.text(0.5, 0.5, "Please check at least one approximation.",
                           ha='center', va='center', transform=ax_2d.transAxes, fontsize=13, color='darkred')
                ax_2d.grid(alpha=0.3)
                fig.canvas.draw_idle()
                return

            if lat_choice == 'Square':
                path_ka, path_x, tick_locs, tick_labels = build_square_bz_path(num_seg=150)
                for mode in active_modes:
                    st = MODE_STYLES[mode]
                    energies = solve_square_path(a_val, path_ka, mode=mode)
                    ax_2d.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
                    ax_2d.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])

                ax_2d.axhline(0.0, color='blue', linestyle=':', alpha=0.35, label=r'$\epsilon_1 = 0$')
                ax_2d.axhline(0.8, color='red', linestyle=':', alpha=0.35, label=r'$\epsilon_2 = 0.8\,t_0$')
                for t in tick_locs:
                    ax_2d.axvline(t, color='gray', linestyle='--', alpha=0.5)

                ax_2d.set_xticks(tick_locs)
                ax_2d.set_xticklabels(tick_labels, fontsize=12)
                ax_2d.set_title(rf'Square Lattice Dispersion: $a = {a_val:.2f}\,d_0$ | $t_0 = {T0:.1f}$',
                                fontsize=13, fontweight='bold')
                if a_val == 5.0:
                    ax_2d.set_ylim(-1.5, 2.5)
                elif a_val >= 7.0:
                    ax_2d.set_ylim(-0.3, 1.5)
            else:
                k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
                for mode in active_modes:
                    st = MODE_STYLES[mode]
                    energies = solve_rectangular_path(a_val, b_val, k_path, mode=mode)
                    ax_2d.plot(path_x, energies[:, 0], color=st['color'], ls=st['ls'], lw=st['lw'], label=st['label'])
                    ax_2d.plot(path_x, energies[:, 1], color=st['color'], ls=st['ls'], lw=st['lw'])
                    ax_2d.plot(path_x, energies[:, 2], color=st['color'], ls=st['ls'], lw=st['lw'])

                ax_2d.axhline(0.0, color='blue', linestyle=':', alpha=0.35, label=r'$\epsilon_1=\epsilon_3=0$')
                ax_2d.axhline(0.8, color='red', linestyle=':', alpha=0.35, label=r'$\epsilon_2=0.8\,t_0$')
                for t in tick_locs:
                    ax_2d.axvline(t, color='gray', linestyle='--', alpha=0.5)

                ax_2d.set_xticks(tick_locs)
                ax_2d.set_xticklabels(tick_labels, fontsize=12)
                ax_2d.set_title(rf'Crystal B (Centered Rectangular): $a = {a_val:.2f}\,d_0$, $b = {b_val:.2f}\,d_0$ | $t_0 = {T0:.1f}$',
                                fontsize=13, fontweight='bold')
                if a_val == 5.0:
                    ax_2d.set_ylim(-2.0, 3.0)
                elif a_val >= 7.0:
                    ax_2d.set_ylim(-0.8, 1.8)

            ax_2d.set_xlabel(r'Wavevector $\vec{k}$ Path', fontsize=11)
            ax_2d.set_ylabel(r'Energy ($E / t_0$)', fontsize=11)
            ax_2d.grid(alpha=0.3)
            ax_2d.legend(loc='upper right', fontsize=9, framealpha=0.92)

        else:
            # 3D Brillouin Zone Surface View
            ax_2d.set_visible(False)
            ax_3d.set_visible(True)
            ax_3d.clear()

            if not active_modes:
                ax_3d.text2D(0.5, 0.5, "Please check at least one approximation.",
                             ha='center', va='center', transform=ax_3d.transAxes, fontsize=13, color='darkred')
                fig.canvas.draw_idle()
                return

            if lat_choice == 'Square':
                KX, KY = build_square_bz_mesh(a_val, grid_pts=25)
                for mode in active_modes:
                    E1, E2 = solve_square_mesh(a_val, KX, KY, mode=mode)
                    col = MODE_STYLES[mode]['color']
                    ax_3d.plot_wireframe(KX * a_val / np.pi, KY * a_val / np.pi, E1,
                                         color=col, lw=0.7, alpha=0.65, label=f'{mode} ($E_1$)')
                    ax_3d.plot_surface(KX * a_val / np.pi, KY * a_val / np.pi, E2,
                                       color=col, alpha=0.35)

                ax_3d.set_xlabel(r'$k_x a / \pi$', fontsize=10)
                ax_3d.set_ylabel(r'$k_y a / \pi$', fontsize=10)
                ax_3d.set_zlabel(r'Energy ($E / t_0$)', fontsize=10)
                ax_3d.set_title(rf'3D BZ Dispersion (Square): $a = {a_val:.2f}\,d_0$', fontsize=12, fontweight='bold')
                if a_val >= 6.5:
                    ax_3d.set_zlim(-0.4, 1.6)
            else:
                KX, KY = build_rectangular_bz_mesh(a_val, b_val, grid_pts=25)
                for mode in active_modes:
                    E1, E2, E3 = solve_rectangular_mesh(a_val, b_val, KX, KY, mode=mode)
                    col = MODE_STYLES[mode]['color']
                    ax_3d.plot_wireframe(KX * a_val / np.pi, KY * b_val / np.pi, E1,
                                         color=col, lw=0.6, alpha=0.6, label=f'{mode} ($E_1$)')
                    ax_3d.plot_surface(KX * a_val / np.pi, KY * b_val / np.pi, E2,
                                       color=col, alpha=0.25)
                    ax_3d.plot_wireframe(KX * a_val / np.pi, KY * b_val / np.pi, E3,
                                         color=col, lw=0.6, alpha=0.6, linestyle=':')

                ax_3d.set_xlabel(r'$k_x a / \pi$', fontsize=10)
                ax_3d.set_ylabel(r'$k_y b / \pi$', fontsize=10)
                ax_3d.set_zlabel(r'Energy ($E / t_0$)', fontsize=10)
                ax_3d.set_title(rf'3D BZ Dispersion (Crystal B): $a = {a_val:.2f}\,d_0$, $b = {b_val:.2f}\,d_0$',
                                fontsize=12, fontweight='bold')
                if a_val >= 6.5:
                    ax_3d.set_zlim(-1.0, 1.8)

            ax_3d.legend(loc='upper left', fontsize=8)

        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    radio_lattice.on_clicked(update)
    radio_view.on_clicked(update)
    chk_approx.on_clicked(update)

    update()
    plt.show()

# ==============================================================================
# Batch Figure Reproduction
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

def reproduce_all_figures():
    """Reproduces all required publication figures for HW2 across all models and lattice constants."""
    print("=" * 72)
    print("Executing Batch Reproduction of all HW2 Figures (User Inverse-Eta Solver)")
    print(f"Parameters: Fixed t_0 = {T0}, d_0 = {D0}")
    print("=" * 72)

    a_values = [5.0, 7.0, 9.0]
    b_ratio = 1.2

    # --- 1. Square Lattice Figures ---
    print("\n[1/2] Processing Square Lattice (Crystal A)...")
    path_ka, path_x, tick_locs, tick_labels = build_square_bz_path(num_seg=150)

    # Individual 3-panel plots for each mode across a in {5, 7, 9}
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=(mode not in ('Full', 'NN only')))
        for idx, a_val in enumerate(a_values):
            ax = axes[idx]
            energies = solve_square_path(a_val, path_ka, mode=mode)
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

    # Detailed per-lattice-constant comparisons
    for a_val in a_values:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for mode in MODES:
            energies = solve_square_path(a_val, path_ka, mode=mode)
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

    # Master 3-panel comparison overview across all a values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for idx, a_val in enumerate(a_values):
        ax = axes[idx]
        for mode in MODES:
            energies = solve_square_path(a_val, path_ka, mode=mode)
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

    # --- 2. Rectangular Lattice Figures ---
    print("\n[2/2] Processing Rectangular Lattice (Crystal B)...")
    for mode in MODES:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
        for idx, a_val in enumerate(a_values):
            b_val = b_ratio * a_val
            ax = axes[idx]
            k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
            energies = solve_rectangular_path(a_val, b_val, k_path, mode=mode)

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

    # Detailed per-lattice-constant comparisons
    for a_val in a_values:
        b_val = b_ratio * a_val
        k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
        fig, ax = plt.subplots(figsize=(11, 6.8))

        for mode in MODES:
            energies = solve_rectangular_path(a_val, b_val, k_path, mode=mode)
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

    # Master 3-panel comparison overview across all a values
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for idx, a_val in enumerate(a_values):
        b_val = b_ratio * a_val
        k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val, num_seg=150)
        ax = axes[idx]

        for mode in MODES:
            energies = solve_rectangular_path(a_val, b_val, k_path, mode=mode)
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

    print("\n[SUCCESS] Batch reproduction complete. All figures updated in target folders.")

# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Homework 2 Tight-Binding Band Structure Calculator & Interactive Explorer"
    )
    parser.add_argument(
        '-i', '--interactive', action='store_true',
        help="Launch interactive GUI explorer with lattice slider, checkboxes, and radio buttons."
    )
    parser.add_argument(
        '-b', '--batch', action='store_true',
        help="Run batch generation to reproduce all figures across all models and lattice constants."
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        # Default action is batch figure reproduction
        matplotlib.use('Agg')
        reproduce_all_figures()

if __name__ == '__main__':
    main()
