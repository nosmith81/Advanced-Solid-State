"""
Interactive electronic band structure calculation for a 2D tight-binding model
on a square lattice with a two-orbital basis.

Includes:
1. Original Formulation: M = eps_bar + t(k) @ inv(eta(k)), solved via np.linalg.eigvals
2. AI Agent Suggestion: Generalized eigensolver with H(k) = eps_bar @ eta(k) + t(k) and S(k) = eta(k)
3. General Generalized Eigensolver: A c = E B c without assuming Hermiticity
4. Symmetrized Hermitian Formulation: H_sym(k) = 0.5*(eps_bar @ eta(k) + eta(k) @ eps_bar) + t(k)
   which guarantees strict Hermiticity and physical stability with scipy.linalg.eigh.

Renamed structure factors from eps_0, eps_1 -> gamma_NN, gamma_NNN to avoid
shadowing atomic on-site energies eps_1, eps_2.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Try to import scipy.linalg if available, otherwise provide exact NumPy equivalents
try:
    import scipy.linalg as sla
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# Model Parameters and Matrix Calculations
# ==============================================================================

def calc_eta_matrix(R, d0=1.0):
    """
    Computes distance-dependent overlap parameters eta_nm(R).
    Returns a 2x2 matrix:
        [eta_11(R), eta_12(R)]
        [eta_12(R), eta_22(R)]
    """
    eta_11 = (1.0 + R / d0 + (R**2) / (3.0 * d0**2)) * np.exp(-R / d0)
    eta_12 = 0.5 * (1.0 + R / d0 + (R**2) / (64.0 * d0**2)) * np.exp(-R / (2.0 * d0)) - 0.5 * np.exp(-R / d0)
    eta_22 = (1.0 + R / (2.0 * d0) + (R**2) / (24.0 * d0**2)) * np.exp(-R / (2.0 * d0))
    return np.array([[eta_11, eta_12],
                     [eta_12, eta_22]], dtype=float)

def calc_t_matrix(R, d0=1.0, t0=1.0):
    """
    Computes distance-dependent hopping parameters t_nm(R).
    Returns a 2x2 matrix:
        [t_11(R), t_12(R)]
        [t_12(R), t_22(R)]
    """
    if np.isclose(R, 0.0):
        return np.zeros((2, 2), dtype=float)
    
    t_11 = -t0 * (1.0 + 2.0 * R / d0) * np.exp(-R / d0)
    t_12 = -t0 * (1.0 + 2.0 * R / d0) * np.exp(-R / d0) + t0 * (1.0 + R / (16.0 * d0)) * np.exp(-R / (2.0 * d0))
    t_22 = -t0 * (1.0 + R / (2.0 * d0) - (R**2) / (24.0 * d0**2)) * np.exp(-R / (2.0 * d0))
    return np.array([[t_11, t_12],
                     [t_12, t_22]], dtype=float)

# ==============================================================================
# Eigensolvers
# ==============================================================================

def solve_generalized_hermitian(H, S, lower=True):
    """
    Solves H c = E S c using Cholesky decomposition of S (equivalent to scipy.linalg.eigh(H, S)).
    Requires S to be positive-definite.
    Note: If H is non-Hermitian, eigh only inspects either the lower or upper triangle!
    """
    if HAS_SCIPY:
        try:
            return sla.eigh(H, S, lower=lower, eigvals_only=True)
        except sla.LinAlgError:
            return np.array([np.nan, np.nan])
    else:
        try:
            L = np.linalg.cholesky(S)
            L_inv = np.linalg.inv(L)
            if lower:
                H_sym = np.tril(H) + np.tril(H, -1).T
            else:
                H_sym = np.triu(H) + np.triu(H, 1).T
            C = L_inv @ H_sym @ L_inv.T
            return np.linalg.eigvalsh(C)
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan])

def solve_generalized_non_hermitian(A, B):
    """
    Solves A c = E B c for general (possibly non-Hermitian) matrices.
    Equivalent to scipy.linalg.eig(A, B).
    For 2x2, solved via closed-form roots of det(A - E*B) = 0.
    """
    if HAS_SCIPY:
        try:
            evals = sla.eig(A, B, right=False)
            return evals
        except sla.LinAlgError:
            pass
    # Closed-form 2x2 roots:
    detB = np.linalg.det(B)
    detA = np.linalg.det(A)
    tr_adj = A[0,0]*B[1,1] + A[1,1]*B[0,0] - A[0,1]*B[1,0] - A[1,0]*B[0,1]
    return np.roots([detB, -tr_adj, detA])

# ==============================================================================
# Band Calculations Along Path
# ==============================================================================

def get_bands_along_path(a_val, kx_a, ky_a, d0=1.0, t0=1.0,
                         mode='NN + NNN', solver='User (Inversion)'):
    """
    Calculates E_1(k) and E_2(k) along the path given dimensionless (kx*a, ky*a).
    
    mode options:
        'NN + NNN'   : Full model (NN + NNN with overlap)
        'NN only'    : Nearest-Neighbor only (with overlap)
        'No Overlap' : NN + NNN with overlap integrals neglected (S = Identity)
        
    solver options:
        'User (Inversion)'      : M = eps_bar + t_k @ inv(eta_k), np.linalg.eigvals
        'Agent (eigh lower)'    : H = eps_bar @ eta_k + t_k, S = eta_k, eigh(lower)
        'Agent (eigh upper)'    : H = eps_bar @ eta_k + t_k, S = eta_k, eigh(upper)
        'GenEig (Non-Hermitian)': A = eps_bar @ eta_k + t_k, B = eta_k, exact geneig
        'Symmetrized (eigh)'    : H = 0.5*(eps_bar@eta + eta@eps_bar) + t, eigh
    """
    eps_bar = np.array([[0.0, 0.0],
                        [0.0, 0.8 * t0]], dtype=float)
    I2 = np.eye(2, dtype=float)
    num_pts = len(kx_a)
    energies = np.zeros((num_pts, 2), dtype=float)

    include_nnn = (mode in ('NN + NNN', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    R_nn = a_val
    t_nn = calc_t_matrix(R_nn, d0=d0, t0=t0)
    eta_nn = calc_eta_matrix(R_nn, d0=d0) if not neglect_overlap else None

    if include_nnn:
        R_nnn = np.sqrt(2.0) * a_val
        t_nnn = calc_t_matrix(R_nnn, d0=d0, t0=t0)
        eta_nnn = calc_eta_matrix(R_nnn, d0=d0) if not neglect_overlap else None

    for i in range(num_pts):
        # Renamed from eps_0, eps_1 to avoid variable shadowing
        gamma_NN = 2.0 * (np.cos(kx_a[i]) + np.cos(ky_a[i]))
        if include_nnn:
            gamma_NNN = 4.0 * np.cos(kx_a[i]) * np.cos(ky_a[i])
            t_k = t_nn * gamma_NN + t_nnn * gamma_NNN
        else:
            t_k = t_nn * gamma_NN

        if neglect_overlap:
            eta_k = I2
        else:
            if include_nnn:
                eta_k = I2 + eta_nn * gamma_NN + eta_nnn * gamma_NNN
            else:
                eta_k = I2 + eta_nn * gamma_NN

        # Solve according to chosen solver
        if solver == 'User (Inversion)' or neglect_overlap:
            if neglect_overlap:
                M = eps_bar + t_k
            else:
                M = eps_bar + t_k @ np.linalg.inv(eta_k)
            evals = np.linalg.eigvals(M)
            energies[i] = np.sort(np.real(evals))

        elif solver == 'Agent (eigh lower)':
            H = eps_bar @ eta_k + t_k
            S = eta_k
            evals = solve_generalized_hermitian(H, S, lower=True)
            energies[i] = np.sort(np.real(evals))

        elif solver == 'Agent (eigh upper)':
            H = eps_bar @ eta_k + t_k
            S = eta_k
            evals = solve_generalized_hermitian(H, S, lower=False)
            energies[i] = np.sort(np.real(evals))

        elif solver == 'GenEig (Non-Hermitian)':
            H = eps_bar @ eta_k + t_k
            S = eta_k
            evals = solve_generalized_non_hermitian(H, S)
            energies[i] = np.sort(np.real(evals))

        elif solver == 'Symmetrized (eigh)':
            H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k
            S = eta_k
            evals = solve_generalized_hermitian(H_sym, S, lower=True)
            energies[i] = np.sort(np.real(evals))

    return energies

# ==============================================================================
# Interactive GUI
# ==============================================================================

def main():
    d0_val = 1.0
    t0_init = 1.0
    a_init = 7.0 * d0_val  # Default to physically valid regime a >= 5.42 d0
    initial_mode = 'NN + NNN'
    initial_solver = 'User (Inversion)'

    num_seg = 150
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)

    path_ka = np.vstack([k1, k2, k3])
    kx_a = path_ka[:, 0]
    ky_a = path_ka[:, 1]

    path_x = np.linspace(0.0, 3.0, len(path_ka))
    tick_locs = [0.0, 1.0, 2.0, 3.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']

    energies_init = get_bands_along_path(a_init, kx_a, ky_a, d0=d0_val, t0=t0_init,
                                         mode=initial_mode, solver=initial_solver)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    plt.subplots_adjust(bottom=0.30, left=0.28)

    line1, = ax.plot(path_x, energies_init[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
    line2, = ax.plot(path_x, energies_init[:, 1], 'r-', lw=2.2, label=r'$E_2(\vec{k})$ (Upper band)')

    ref_eps1 = ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = 0$')
    ref_eps2 = ax.axhline(0.8 * t0_init, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8 t_0$')

    for t in tick_locs:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.6)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_ylabel(r'Energy ($E / t_0$)', fontsize=13)
    ax.set_xlabel(r'Wavevector $\vec{k}$ path', fontsize=12)
    title_text = ax.set_title(f'Square Lattice Band Structure [{initial_mode}] | Solver: {initial_solver}\n$a = {a_init/d0_val:.2f} d_0$, $t_0 = {t0_init:.2f}$', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)

    y_min = np.nanmin(energies_init) - 0.2
    y_max = np.nanmax(energies_init) + 0.2
    ax.set_ylim(y_min, y_max)

    # Radio buttons for Model
    ax_radio_mode = plt.axes([0.03, 0.60, 0.20, 0.22])
    radio_mode = RadioButtons(ax_radio_mode, ('NN + NNN', 'NN only', 'No Overlap'), active=0)
    ax_radio_mode.set_title("Model Approximation", fontsize=10, fontweight='bold')

    # Radio buttons for Solver
    ax_radio_solver = plt.axes([0.03, 0.15, 0.20, 0.38])
    solvers = ('User (Inversion)', 'Agent (eigh lower)', 'Agent (eigh upper)',
               'GenEig (Non-Hermitian)', 'Symmetrized (eigh)')
    radio_solver = RadioButtons(ax_radio_solver, solvers, active=0)
    ax_radio_solver.set_title("Solver Method", fontsize=10, fontweight='bold')

    # Slider 1: Lattice parameter a (units of d0)
    ax_a = plt.axes([0.38, 0.12, 0.54, 0.03])
    a_slider = Slider(
        ax=ax_a,
        label=r'Lattice constant $a / d_0$',
        valmin=3.2,
        valmax=12.0,
        valinit=a_init / d0_val,
        valstep=0.05
    )

    # Slider 2: Energy parameter t_0
    ax_t0 = plt.axes([0.38, 0.05, 0.54, 0.03])
    t0_slider = Slider(
        ax=ax_t0,
        label=r'Energy unit $t_0$',
        valmin=0.1,
        valmax=4.0,
        valinit=t0_init,
        valstep=0.05
    )

    def update(val=None):
        a_current = a_slider.val * d0_val
        t0_current = t0_slider.val
        mode_current = radio_mode.value_selected
        solver_current = radio_solver.value_selected

        new_energies = get_bands_along_path(a_current, kx_a, ky_a, d0=d0_val, t0=t0_current,
                                            mode=mode_current, solver=solver_current)

        line1.set_ydata(new_energies[:, 0])
        line2.set_ydata(new_energies[:, 1])

        ref_eps2.set_ydata([0.8 * t0_current, 0.8 * t0_current])
        title_text.set_text(f'Square Lattice Band Structure [{mode_current}] | Solver: {solver_current}\n$a = {a_current/d0_val:.2f} d_0$, $t_0 = {t0_current:.2f}$')

        valid_energies = new_energies[~np.isnan(new_energies)]
        if len(valid_energies) > 0:
            cur_min = np.min(valid_energies) - 0.15 * t0_current
            cur_max = np.max(valid_energies) + 0.15 * t0_current
            # Clamp limits if singular
            cur_min = max(cur_min, -50.0)
            cur_max = min(cur_max, 300.0)
            ax.set_ylim(cur_min, cur_max)

        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    t0_slider.on_changed(update)
    radio_mode.on_clicked(update)
    radio_solver.on_clicked(update)

    plt.show()

if __name__ == '__main__':
    main()
