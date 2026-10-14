"""
Interactive electronic band structure calculation for Crystal B:
Centered Rectangular Lattice with a 3-orbital basis:
  - Orbital 1 (Site 1, (0, 0)): atomic energy eps_1 = 0
  - Orbital 2 (Site 1, (0, 0)): atomic energy eps_2 = 0.8 * t_0
  - Orbital 3 (Site 2, (a/2, b/2)): atomic energy eps_3 = eps_1 = 0

Path along high-symmetry BZ lines for rectangular lattice:
  Gamma (0, 0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0, 0)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ==============================================================================
# Analytical Distance-Dependent Overlap and Hopping Integrals
# ==============================================================================

def calc_eta_matrix(R, d0=1.0):
    """
    Computes distance-dependent overlap parameters eta_nm(R).
    Returns 2x2 matrix for orbital types (1 and 2):
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
    Returns 2x2 matrix for orbital types (1 and 2):
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
# 3x3 Matrix Assembly and Band Calculation
# ==============================================================================

def get_crystal_b_bands(a_val, b_val, k_path, d0=1.0, t0=1.0,
                        mode='NN + NNN', solver='Symmetrized (eigh)'):
    """
    Calculates E_1(k), E_2(k), E_3(k) for Crystal B along k_path = (kx, ky).
    
    Basis indexing:
      1: Site 1 (0, 0), Orbital 1 (eps_1 = 0)
      2: Site 1 (0, 0), Orbital 2 (eps_2 = 0.8*t0)
      3: Site 2 (a/2, b/2), Orbital 1 (eps_1 = 0)
    """
    num_pts = len(k_path)
    energies = np.zeros((num_pts, 3), dtype=float)

    eps_bar = np.diag([0.0, 0.8 * t0, 0.0])
    I3 = np.eye(3, dtype=float)

    include_nnn = (mode in ('NN + NNN', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    # Distance norms
    # Intra-sublattice
    R_a = a_val
    R_b = b_val
    R_ab = np.sqrt(a_val**2 + b_val**2)

    # Inter-sublattice (vector addition: d_1 = (a/2, b/2))
    R_inter1 = 0.5 * R_ab

    # Evaluate overlap and hopping matrices for relevant distances
    eta_a = calc_eta_matrix(R_a, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_a = calc_t_matrix(R_a, d0=d0, t0=t0)
    eta_b = calc_eta_matrix(R_b, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_b = calc_t_matrix(R_b, d0=d0, t0=t0)

    if include_nnn:
        eta_ab = calc_eta_matrix(R_ab, d0=d0) if not neglect_overlap else np.zeros((2, 2))
        t_ab = calc_t_matrix(R_ab, d0=d0, t0=t0)

    # Inter-sublattice nearest neighbors
    eta_inter1 = calc_eta_matrix(R_inter1, d0=d0) if not neglect_overlap else np.zeros((2, 2))
    t_inter1 = calc_t_matrix(R_inter1, d0=d0, t0=t0)

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
            # On-site overlap
            eta_k[0, 0] = 1.0
            eta_k[1, 1] = 1.0
            eta_k[2, 2] = 1.0

        # --- Intra-sublattice terms ---
        # Block (1, 1): orbital 1 on site 1
        if not neglect_overlap:
            eta_k[0, 0] += 2.0*eta_a[0,0]*cos_ax + 2.0*eta_b[0,0]*cos_by + (4.0*eta_ab[0,0]*cos_diag if include_nnn else 0.0)
        t_k[0, 0] = 2.0*t_a[0,0]*cos_ax + 2.0*t_b[0,0]*cos_by + (4.0*t_ab[0,0]*cos_diag if include_nnn else 0.0)

        # Block (2, 2): orbital 2 on site 1
        if not neglect_overlap:
            eta_k[1, 1] += 2.0*eta_a[1,1]*cos_ax + 2.0*eta_b[1,1]*cos_by + (4.0*eta_ab[1,1]*cos_diag if include_nnn else 0.0)
        t_k[1, 1] = 2.0*t_a[1,1]*cos_ax + 2.0*t_b[1,1]*cos_by + (4.0*t_ab[1,1]*cos_diag if include_nnn else 0.0)

        # Block (1, 2) & (2, 1): orbital 1 & 2 on site 1
        val_eta_12 = 2.0*eta_a[0,1]*cos_ax + 2.0*eta_b[0,1]*cos_by + (4.0*eta_ab[0,1]*cos_diag if include_nnn else 0.0) if not neglect_overlap else 0.0
        val_t_12 = 2.0*t_a[0,1]*cos_ax + 2.0*t_b[0,1]*cos_by + (4.0*t_ab[0,1]*cos_diag if include_nnn else 0.0)
        eta_k[0, 1] = eta_k[1, 0] = val_eta_12
        t_k[0, 1] = t_k[1, 0] = val_t_12

        # Block (3, 3): orbital 1 on site 2 (identical intra-sublattice environment as (1, 1))
        eta_k[2, 2] = eta_k[0, 0]
        t_k[2, 2] = t_k[0, 0]

        # --- Inter-sublattice terms (Site 1 <-> Site 2) ---
        # (1, 3): orb 1 on site 1 <-> orb 1 on site 2 (uses eta_11, t_11)
        val_eta_13 = eta_inter1[0, 0] * cos_inter if not neglect_overlap else 0.0
        val_t_13 = t_inter1[0, 0] * cos_inter
        eta_k[0, 2] = eta_k[2, 0] = val_eta_13
        t_k[0, 2] = t_k[2, 0] = val_t_13

        # (2, 3): orb 2 on site 1 <-> orb 1 on site 2 (uses eta_12, t_12)
        val_eta_23 = eta_inter1[0, 1] * cos_inter if not neglect_overlap else 0.0
        val_t_23 = t_inter1[0, 1] * cos_inter
        eta_k[1, 2] = eta_k[2, 1] = val_eta_23
        t_k[1, 2] = t_k[2, 1] = val_t_23

        # --- Solve eigenvalue problem ---
        if neglect_overlap:
            M = eps_bar + t_k
            evals = np.linalg.eigvalsh(M)
            energies[i] = np.sort(evals)

        elif solver == 'Symmetrized (eigh)':
            # H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k
            H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k
            try:
                L = np.linalg.cholesky(eta_k)
                L_inv = np.linalg.inv(L)
                C = L_inv @ H_sym @ L_inv.T
                evals = np.linalg.eigvalsh(C)
                energies[i] = np.sort(evals)
            except np.linalg.LinAlgError:
                # Fallback if indefinite
                energies[i] = np.array([np.nan, np.nan, np.nan])

        elif solver == 'Inversion (M = eps + t @ inv(eta))':
            try:
                M = eps_bar + t_k @ np.linalg.inv(eta_k)
                evals = np.linalg.eigvals(M)
                energies[i] = np.sort(np.real(evals))
            except np.linalg.LinAlgError:
                energies[i] = np.array([np.nan, np.nan, np.nan])

    return energies

# ==============================================================================
# Interactive GUI
# ==============================================================================

def main():
    d0_val = 1.0
    t0_init = 1.0
    a_init = 7.0 * d0_val
    b_init = 8.5 * d0_val
    initial_mode = 'NN + NNN'
    initial_solver = 'Symmetrized (eigh)'

    # High-symmetry path: Gamma (0,0) -> X (pi/a, 0) -> M (pi/a, pi/b) -> Y (0, pi/b) -> Gamma (0,0)
    num_seg = 100
    def build_k_path(a, b):
        k_G = np.array([0.0, 0.0])
        k_X = np.array([np.pi / a, 0.0])
        k_M = np.array([np.pi / a, np.pi / b])
        k_Y = np.array([0.0, np.pi / b])

        k1 = np.linspace(k_G, k_X, num_seg, endpoint=False)
        k2 = np.linspace(k_X, k_M, num_seg, endpoint=False)
        k3 = np.linspace(k_M, k_Y, num_seg, endpoint=False)
        k4 = np.linspace(k_Y, k_G, num_seg + 1, endpoint=True)

        return np.vstack([k1, k2, k3, k4])

    k_path = build_k_path(a_init, b_init)
    path_x = np.linspace(0.0, 4.0, len(k_path))
    tick_locs = [0.0, 1.0, 2.0, 3.0, 4.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r'$\Gamma$']

    energies_init = get_crystal_b_bands(a_init, b_init, k_path, d0=d0_val, t0=t0_init,
                                        mode=initial_mode, solver=initial_solver)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    plt.subplots_adjust(bottom=0.32, left=0.28)

    line1, = ax.plot(path_x, energies_init[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
    line2, = ax.plot(path_x, energies_init[:, 1], 'g-', lw=2.2, label=r'$E_2(\vec{k})$ (Middle band)')
    line3, = ax.plot(path_x, energies_init[:, 2], 'r-', lw=2.2, label=r'$E_3(\vec{k})$ (Upper band)')

    # Atomic reference levels
    ref_eps1 = ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = \epsilon_3 = 0$')
    ref_eps2 = ax.axhline(0.8 * t0_init, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8 t_0$')

    for t in tick_locs:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.6)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=13)
    ax.set_xlabel(r'Wavevector $\vec{k}$ path', fontsize=12)
    title_text = ax.set_title(
        f'Crystal B Band Structure (Centered Rectangular, $3\\times 3$)\n[{initial_mode}] | Solver: {initial_solver} | $a = {a_init/d0_val:.1f}d_0, b = {b_init/d0_val:.1f}d_0, t_0 = {t0_init:.2f}$',
        fontsize=11
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9.5)

    y_min = np.nanmin(energies_init) - 0.2
    y_max = np.nanmax(energies_init) + 0.2
    ax.set_ylim(y_min, y_max)

    # Radio buttons for Model
    ax_radio_mode = plt.axes([0.03, 0.58, 0.20, 0.22])
    radio_mode = RadioButtons(ax_radio_mode, ('NN + NNN', 'NN only', 'No Overlap'), active=0)
    ax_radio_mode.set_title("Model Approximation", fontsize=10, fontweight='bold')

    # Radio buttons for Solver
    ax_radio_solver = plt.axes([0.03, 0.22, 0.20, 0.25])
    solvers = ('Symmetrized (eigh)', 'Inversion (M = eps + t @ inv(eta))')
    radio_solver = RadioButtons(ax_radio_solver, solvers, active=0)
    ax_radio_solver.set_title("Solver Method", fontsize=10, fontweight='bold')

    # Slider 1: Lattice parameter a (units of d0)
    ax_a = plt.axes([0.38, 0.16, 0.54, 0.03])
    a_slider = Slider(ax=ax_a, label=r'Lattice constant $a / d_0$', valmin=4.0, valmax=12.0, valinit=a_init/d0_val, valstep=0.1)

    # Slider 2: Lattice parameter b (units of d0)
    ax_b = plt.axes([0.38, 0.10, 0.54, 0.03])
    b_slider = Slider(ax=ax_b, label=r'Lattice constant $b / d_0$', valmin=4.0, valmax=14.0, valinit=b_init/d0_val, valstep=0.1)

    # Slider 3: Energy parameter t_0
    ax_t0 = plt.axes([0.38, 0.04, 0.54, 0.03])
    t0_slider = Slider(ax=ax_t0, label=r'Energy unit $t_0$', valmin=0.1, valmax=3.0, valinit=t0_init, valstep=0.05)

    def update(val=None):
        a_cur = a_slider.val * d0_val
        b_cur = b_slider.val * d0_val
        t0_cur = t0_slider.val
        mode_cur = radio_mode.value_selected
        solver_cur = radio_solver.value_selected

        current_k_path = build_k_path(a_cur, b_cur)
        new_energies = get_crystal_b_bands(a_cur, b_cur, current_k_path, d0=d0_val, t0=t0_cur,
                                           mode=mode_cur, solver=solver_cur)

        line1.set_ydata(new_energies[:, 0])
        line2.set_ydata(new_energies[:, 1])
        line3.set_ydata(new_energies[:, 2])

        ref_eps2.set_ydata([0.8 * t0_cur, 0.8 * t0_cur])
        title_text.set_text(
            f'Crystal B Band Structure (Centered Rectangular, $3\\times 3$)\n[{mode_cur}] | Solver: {solver_cur} | $a = {a_cur/d0_val:.1f}d_0, b = {b_cur/d0_val:.1f}d_0, t_0 = {t0_cur:.2f}$'
        )

        valid = new_energies[~np.isnan(new_energies)]
        if len(valid) > 0:
            cur_min = np.min(valid) - 0.15 * t0_cur
            cur_max = np.max(valid) + 0.15 * t0_cur
            cur_min = max(cur_min, -50.0)
            cur_max = min(cur_max, 300.0)
            ax.set_ylim(cur_min, cur_max)

        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    b_slider.on_changed(update)
    t0_slider.on_changed(update)
    radio_mode.on_clicked(update)
    radio_solver.on_clicked(update)

    plt.show()

if __name__ == '__main__':
    main()
