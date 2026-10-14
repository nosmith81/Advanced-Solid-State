"""
Lattice_B.py: Electronic band structure calculation for Crystal B (Centered Rectangular Lattice).

Model Architecture:
- 2D Centered Rectangular Lattice with primitive rectangular parameters a (along x) and b (along y).
- 2-site conventional basis:
    * Site 1 at (0, 0): hosts 2 atomic orbitals (phi_1 and phi_2)
    * Site 2 at (a/2, b/2): hosts 1 atomic orbital (phi_1)
- Basis states (3-orbital system):
    * State 1: Site 1, Orbital 1 (reference energy eps_1 = 0)
    * State 2: Site 1, Orbital 2 (reference energy eps_2 = 0.8 * t_0)
    * State 3: Site 2, Orbital 1 (reference energy eps_3 = eps_1 = 0)

Numerical Stability:
- Employs the manifestly Hermitian symmetrized Hamiltonian:
    H_sym(k) = 0.5 * [eps_bar @ eta(k) + eta(k) @ eps_bar] + t(k)
- Solved via generalized Hermitian eigensolver (spectral transformation S^(-1/2) H_sym S^(-1/2))
  to guarantee strictly real eigenvalues and eliminate numerical artifacts.

Produces:
1. band_structures_lattice_b.png: Band structures for a = 5*d0, 7*d0, 9*d0 (NN + NNN with Overlap)
2. band_structure_comparisons_lattice_b.png: Comparison between Full, NN-only, and No Overlap models
3. Interactive GUI (when run interactively) with sliders for a, b, t_0 and radio buttons for approximations.
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ==============================================================================
# Distance-Dependent Overlap and Hopping Functions
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
# Numerically Stable Generalized Hermitian Eigensolver
# ==============================================================================

def solve_generalized_hermitian_spectral(H_sym, S, tol=1e-10):
    """
    Solves H_sym c = E S c for real symmetric H_sym and S.
    Uses spectral decomposition S = V Lambda V^T:
      S^(-1/2) = V Lambda^(-1/2) V^T
      C = S^(-1/2) H_sym S^(-1/2)
      E = eigvalsh(C)
    Guarantees strictly real eigenvalues. If S is not positive-definite,
    returns NaNs to signal the physical breakdown of the truncated overlap model.
    """
    w, V = np.linalg.eigh(S)
    if np.all(w > tol):
        S_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        C = S_inv_sqrt @ H_sym @ S_inv_sqrt
        # Enforce exact symmetry
        C_sym = 0.5 * (C + C.T)
        return np.sort(np.linalg.eigvalsh(C_sym))
    else:
        # Fallback using regularized inverse / sign to prevent unhandled crash
        # while preserving real eigenvalues
        w_reg = np.maximum(w, tol)
        S_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w_reg)) @ V.T
        C = S_inv_sqrt @ H_sym @ S_inv_sqrt
        C_sym = 0.5 * (C + C.T)
        evals = np.sort(np.linalg.eigvalsh(C_sym))
        # Mask points where negative overlap occurs
        if np.any(w <= 0):
            evals = np.array([np.nan, np.nan, np.nan])
        return evals

# ==============================================================================
# 3-Band Dispersion Calculation
# ==============================================================================

def compute_bands_k_path(a_val, b_val, k_path, d0=1.0, t0=1.0, mode='NN + NNN'):
    """
    Calculates E_1(k), E_2(k), E_3(k) along k_path for Crystal B.
    
    Modes:
      'NN + NNN'   : Full model (NN + NNN with overlap)
      'NN only'    : Nearest-neighbor only (with overlap)
      'No Overlap' : NN + NNN with overlap integrals neglected (S = Identity)
    """
    num_pts = len(k_path)
    energies = np.zeros((num_pts, 3), dtype=float)

    eps_bar = np.diag([0.0, 0.8 * t0, 0.0])

    include_nnn = (mode in ('NN + NNN', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    # True Euclidean distances
    R_a = a_val
    R_b = b_val
    R_ab = np.sqrt(a_val**2 + b_val**2)
    R_inter = 0.5 * R_ab  # Distance from (0,0) to (+-a/2, +-b/2)

    # Overlap and hopping matrices
    eta_a = calc_eta_matrix(R_a, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_a = calc_t_matrix(R_a, d0=d0, t0=t0)

    eta_b = calc_eta_matrix(R_b, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_b = calc_t_matrix(R_b, d0=d0, t0=t0)

    if include_nnn:
        eta_ab = calc_eta_matrix(R_ab, d0=d0) if not neglect_overlap else np.zeros((2, 2))
        t_ab = calc_t_matrix(R_ab, d0=d0, t0=t0)

    eta_inter = calc_eta_matrix(R_inter, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_inter = calc_t_matrix(R_inter, d0=d0, t0=t0)

    for i in range(num_pts):
        kx, ky = k_path[i]

        # Phase factors
        cos_ax = np.cos(kx * a_val)
        cos_by = np.cos(ky * b_val)
        cos_diag = np.cos(kx * a_val) * np.cos(ky * b_val) if include_nnn else 0.0
        cos_inter = 4.0 * np.cos(0.5 * kx * a_val) * np.cos(0.5 * ky * b_val)

        eta_k = np.zeros((3, 3), dtype=float)
        t_k = np.zeros((3, 3), dtype=float)

        if not neglect_overlap:
            eta_k[0, 0] = 1.0
            eta_k[1, 1] = 1.0
            eta_k[2, 2] = 1.0

        # --- Intra-sublattice (Site 1 <-> Site 1, and Site 2 <-> Site 2) ---
        # Block (1, 1): orbital 1 on site 1
        if not neglect_overlap:
            eta_k[0, 0] += 2.0*eta_a[0,0]*cos_ax + 2.0*eta_b[0,0]*cos_by + (4.0*eta_ab[0,0]*cos_diag if include_nnn else 0.0)
        t_k[0, 0] = 2.0*t_a[0,0]*cos_ax + 2.0*t_b[0,0]*cos_by + (4.0*t_ab[0,0]*cos_diag if include_nnn else 0.0)

        # Block (2, 2): orbital 2 on site 1
        if not neglect_overlap:
            eta_k[1, 1] += 2.0*eta_a[1,1]*cos_ax + 2.0*eta_b[1,1]*cos_by + (4.0*eta_ab[1,1]*cos_diag if include_nnn else 0.0)
        t_k[1, 1] = 2.0*t_a[1,1]*cos_ax + 2.0*t_b[1,1]*cos_by + (4.0*t_ab[1,1]*cos_diag if include_nnn else 0.0)

        # Block (1, 2) & (2, 1): orbital 1 <-> 2 on site 1
        val_eta_12 = 2.0*eta_a[0,1]*cos_ax + 2.0*eta_b[0,1]*cos_by + (4.0*eta_ab[0,1]*cos_diag if include_nnn else 0.0) if not neglect_overlap else 0.0
        val_t_12 = 2.0*t_a[0,1]*cos_ax + 2.0*t_b[0,1]*cos_by + (4.0*t_ab[0,1]*cos_diag if include_nnn else 0.0)
        eta_k[0, 1] = eta_k[1, 0] = val_eta_12
        t_k[0, 1] = t_k[1, 0] = val_t_12

        # Block (3, 3): orbital 1 on site 2 (identical intra-sublattice environment as (1, 1))
        eta_k[2, 2] = eta_k[0, 0]
        t_k[2, 2] = t_k[0, 0]

        # --- Inter-sublattice (Site 1 <-> Site 2) ---
        # (1, 3): orb 1 on site 1 <-> orb 1 on site 2 (uses eta_11, t_11)
        val_eta_13 = eta_inter[0, 0] * cos_inter if not neglect_overlap else 0.0
        val_t_13 = t_inter[0, 0] * cos_inter
        eta_k[0, 2] = eta_k[2, 0] = val_eta_13
        t_k[0, 2] = t_k[2, 0] = val_t_13

        # (2, 3): orb 2 on site 1 <-> orb 1 on site 2 (uses eta_12, t_12)
        val_eta_23 = eta_inter[0, 1] * cos_inter if not neglect_overlap else 0.0
        val_t_23 = t_inter[0, 1] * cos_inter
        eta_k[1, 2] = eta_k[2, 1] = val_eta_23
        t_k[1, 2] = t_k[2, 1] = val_t_23

        # --- Solve eigenvalue problem ---
        if neglect_overlap:
            M = eps_bar + t_k
            energies[i] = np.sort(np.linalg.eigvalsh(M))
        else:
            # Symmetrized Hermitian Hamiltonian
            H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k
            energies[i] = solve_generalized_hermitian_spectral(H_sym, eta_k)

    return energies

def build_rectangular_bz_path(a, b, num_seg=150):
    """
    Constructs high-symmetry path:
      Gamma (0, 0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0, 0)
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
# Plot Generation Functions (Matching Part 1 Output Style)
# ==============================================================================

def generate_part1_plot(b_ratio=1.2, save_path='band_structures_lattice_b.png'):
    """
    Produces the 3-panel band structure figure for a = 5*d0, 7*d0, 9*d0
    (NN + NNN with Overlap), matching the layout of band_structures_part1.png.
    """
    a_values = [5.0, 7.0, 9.0]
    t0_val = 1.0
    d0_val = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, a_val in enumerate(a_values):
        b_val = b_ratio * a_val
        ax = axes[idx]
        k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val)
        energies = compute_bands_k_path(a_val, b_val, k_path, d0=d0_val, t0=t0_val, mode='NN + NNN')

        ax.plot(path_x, energies[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
        ax.plot(path_x, energies[:, 1], 'g-', lw=2.2, label=r'$E_2(\vec{k})$ (Middle band)')
        ax.plot(path_x, energies[:, 2], 'r-', lw=2.2, label=r'$E_3(\vec{k})$ (Upper band)')

        # Atomic reference levels
        ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = \epsilon_3 = 0$')
        ax.axhline(0.8 * t0_val, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8 t_0$')

        for t in tick_locs:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=13)
        ax.set_title(f'$a = {int(a_val)}d_0, b = {b_val:.1f}d_0$', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'Wavevector $\vec{k}$', fontsize=12)
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=13)
            ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    plt.suptitle(r'Crystal B: Centered Rectangular Lattice ($3\times 3$) [NN + NNN with Overlap]', fontsize=14, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Generated: {save_path}")
    plt.close()

def generate_comparisons_plot(a_val=7.0, b_ratio=1.2, save_path='band_structure_comparisons_lattice_b.png'):
    """
    Produces the comparison figure comparing Full vs NN-only vs No Overlap,
    matching the layout of band_structure_comparisons.png.
    """
    b_val = b_ratio * a_val
    t0_val = 1.0
    d0_val = 1.0

    k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_val, b_val)

    energies_full = compute_bands_k_path(a_val, b_val, k_path, d0=d0_val, t0=t0_val, mode='NN + NNN')
    energies_nn = compute_bands_k_path(a_val, b_val, k_path, d0=d0_val, t0=t0_val, mode='NN only')
    energies_no_ov = compute_bands_k_path(a_val, b_val, k_path, d0=d0_val, t0=t0_val, mode='No Overlap')

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot Full model
    ax.plot(path_x, energies_full[:, 0], 'b-', lw=2.2, label='Full (NN + NNN + Overlap)')
    ax.plot(path_x, energies_full[:, 1], 'b-', lw=2.2)
    ax.plot(path_x, energies_full[:, 2], 'b-', lw=2.2)

    # Plot NN only
    ax.plot(path_x, energies_nn[:, 0], 'g--', lw=2.0, label='NN only (with Overlap)')
    ax.plot(path_x, energies_nn[:, 1], 'g--', lw=2.0)
    ax.plot(path_x, energies_nn[:, 2], 'g--', lw=2.0)

    # Plot No Overlap
    ax.plot(path_x, energies_no_ov[:, 0], 'r:', lw=2.2, label='No Overlap (NN + NNN)')
    ax.plot(path_x, energies_no_ov[:, 1], 'r:', lw=2.2)
    ax.plot(path_x, energies_no_ov[:, 2], 'r:', lw=2.2)

    for t in tick_locs:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=13)
    ax.set_xlabel(r'Wavevector $\vec{k}$', fontsize=12)
    ax.set_title(f'Crystal B: Comparison of Approximations ($a = {a_val:.1f}d_0, b = {b_val:.1f}d_0$)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Generated: {save_path}")
    plt.close()

# ==============================================================================
# Interactive GUI Mode
# ==============================================================================

def run_interactive_gui():
    """Launches interactive window with sliders and radio toggles."""
    d0_val = 1.0
    t0_init = 1.0
    a_init = 7.0 * d0_val
    b_init = 8.5 * d0_val
    initial_mode = 'NN + NNN'

    k_path, path_x, tick_locs, tick_labels = build_rectangular_bz_path(a_init, b_init)
    energies_init = compute_bands_k_path(a_init, b_init, k_path, d0=d0_val, t0=t0_init, mode=initial_mode)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    plt.subplots_adjust(bottom=0.32, left=0.28)

    line1, = ax.plot(path_x, energies_init[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
    line2, = ax.plot(path_x, energies_init[:, 1], 'g-', lw=2.2, label=r'$E_2(\vec{k})$ (Middle band)')
    line3, = ax.plot(path_x, energies_init[:, 2], 'r-', lw=2.2, label=r'$E_3(\vec{k})$ (Upper band)')

    ref_eps1 = ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = \epsilon_3 = 0$')
    ref_eps2 = ax.axhline(0.8 * t0_init, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8 t_0$')

    for t in tick_locs:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=13)
    ax.set_xlabel(r'Wavevector $\vec{k}$ path', fontsize=12)
    title_text = ax.set_title(
        f'Crystal B Band Structure (Centered Rectangular, $3\\times 3$)\n[{initial_mode}] | $a = {a_init/d0_val:.1f}d_0, b = {b_init/d0_val:.1f}d_0, t_0 = {t0_init:.2f}$',
        fontsize=12
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    y_min = np.nanmin(energies_init) - 0.2
    y_max = np.nanmax(energies_init) + 0.2
    ax.set_ylim(y_min, y_max)

    # Radio buttons
    ax_radio_mode = plt.axes([0.03, 0.45, 0.20, 0.25])
    radio_mode = RadioButtons(ax_radio_mode, ('NN + NNN', 'NN only', 'No Overlap'), active=0)
    ax_radio_mode.set_title("Approximation", fontsize=10, fontweight='bold')

    # Sliders
    ax_a = plt.axes([0.38, 0.16, 0.54, 0.03])
    a_slider = Slider(ax=ax_a, label=r'Lattice constant $a / d_0$', valmin=4.0, valmax=12.0, valinit=a_init/d0_val, valstep=0.1)

    ax_b = plt.axes([0.38, 0.10, 0.54, 0.03])
    b_slider = Slider(ax=ax_b, label=r'Lattice constant $b / d_0$', valmin=4.0, valmax=14.0, valinit=b_init/d0_val, valstep=0.1)

    ax_t0 = plt.axes([0.38, 0.04, 0.54, 0.03])
    t0_slider = Slider(ax=ax_t0, label=r'Energy unit $t_0$', valmin=0.1, valmax=3.0, valinit=t0_init, valstep=0.05)

    def update(val=None):
        a_cur = a_slider.val * d0_val
        b_cur = b_slider.val * d0_val
        t0_cur = t0_slider.val
        mode_cur = radio_mode.value_selected

        current_k_path, _, _, _ = build_rectangular_bz_path(a_cur, b_cur)
        new_energies = compute_bands_k_path(a_cur, b_cur, current_k_path, d0=d0_val, t0=t0_cur, mode=mode_cur)

        line1.set_ydata(new_energies[:, 0])
        line2.set_ydata(new_energies[:, 1])
        line3.set_ydata(new_energies[:, 2])

        ref_eps2.set_ydata([0.8 * t0_cur, 0.8 * t0_cur])
        title_text.set_text(
            f'Crystal B Band Structure (Centered Rectangular, $3\\times 3$)\n[{mode_cur}] | $a = {a_cur/d0_val:.1f}d_0, b = {b_cur/d0_val:.1f}d_0, t_0 = {t0_cur:.2f}$'
        )

        valid = new_energies[~np.isnan(new_energies)]
        if len(valid) > 0:
            cur_min = np.min(valid) - 0.15 * t0_cur
            cur_max = np.max(valid) + 0.15 * t0_cur
            ax.set_ylim(cur_min, cur_max)

        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    b_slider.on_changed(update)
    t0_slider.on_changed(update)
    radio_mode.on_clicked(update)

    plt.show()

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == '__main__':
    # Determine execution directory for saving plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hw2_dir = script_dir
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Always generate the static publication plots
    part1_out = os.path.join(repo_root, 'band_structures_lattice_b.png')
    comp_out = os.path.join(repo_root, 'band_structure_comparisons_lattice_b.png')

    print(f"Generating static plots for Crystal B matching Part 1 style...")
    generate_part1_plot(b_ratio=1.2, save_path=part1_out)
    generate_comparisons_plot(a_val=7.0, b_ratio=1.2, save_path=comp_out)

    # If --interactive or -i is passed, launch GUI
    if '--interactive' in sys.argv or '-i' in sys.argv:
        run_interactive_gui()
    else:
        print("To launch the interactive GUI with sliders, run: python3 Lattice_B.py --interactive")
