"""
Interactive electronic band structure calculation for a 2D tight-binding model
on a square lattice with a two-orbital basis.

Features:
- Single main plot with RadioButtons to switch between:
    * 'NN + NNN'     : Full model (Nearest + Next-Nearest Neighbor with overlap)
    * 'NN only'      : Nearest-Neighbor only (with overlap) - Part 2(i)
    * 'No Overlap'   : Neglecting overlap integrals (eta_nm(R) = delta_nm * delta_R,0) - Part 2(ii)
- Interactive slider for lattice parameter 'a' (in units of d_0)
- Interactive slider for energy parameter 't_0'
- On-site atomic reference energies: eps_1 = 0, eps_2 = 0.8 * t_0
- Path along high-symmetry BZ lines: Gamma -> X -> M -> Gamma
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

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

def get_bands_along_path(a_val, kx_a, ky_a, d0=1.0, t0=1.0, mode='NN + NNN'):
    """
    Calculates E_1(k) and E_2(k) along the path given dimensionless (kx*a, ky*a).
    
    mode options:
        'NN + NNN'   : Full model (NN + NNN with overlap)
        'NN only'    : Nearest-Neighbor only (with overlap)
        'No Overlap' : NN + NNN with overlap integrals neglected (eta(k) = Identity)
    """
    eps_bar = np.array([[0.0, 0.0],
                        [0.0, 0.8 * t0]], dtype=float)

    I2 = np.eye(2, dtype=float)
    num_pts = len(kx_a)
    energies = np.zeros((num_pts, 2), dtype=float)

    include_nnn = (mode in ('NN + NNN', 'No Overlap'))
    neglect_overlap = (mode == 'No Overlap')

    # Distance-dependent matrices
    R_nn = a_val
    t_nn = calc_t_matrix(R_nn, d0=d0, t0=t0)
    eta_nn = calc_eta_matrix(R_nn, d0=d0) if not neglect_overlap else None

    if include_nnn:
        R_nnn = np.sqrt(2.0) * a_val
        t_nnn = calc_t_matrix(R_nnn, d0=d0, t0=t0)
        eta_nnn = calc_eta_matrix(R_nnn, d0=d0) if not neglect_overlap else None

    for i in range(num_pts):
        eps_0 = np.cos(kx_a[i]) + np.cos(ky_a[i])

        if include_nnn:
            eps_1 = 2.0 * np.cos(kx_a[i]) * np.cos(ky_a[i])
            t_k = 2.0 * t_nn * eps_0 + 2.0 * t_nnn * eps_1
        else:
            t_k = 2.0 * t_nn * eps_0

        if neglect_overlap:
            # Overlap matrix is simply the Identity matrix
            M = eps_bar + t_k
        else:
            # Include spatial overlap
            if include_nnn:
                eta_k = I2 + 2.0 * eta_nn * eps_0 + 2.0 * eta_nnn * eps_1
            else:
                eta_k = I2 + 2.0 * eta_nn * eps_0
            M = eps_bar + t_k @ np.linalg.inv(eta_k)

        evals = np.linalg.eigvals(M)
        energies[i] = np.sort(np.real(evals))

    return energies

# ==============================================================================
# Main Interactive Application
# ==============================================================================

def main():
    d0_val = 1.0
    t0_init = 1.0
    a_init = 5.0 * d0_val
    initial_mode = 'NN + NNN'

    # Dimensionless k*a along Gamma -> X -> M -> Gamma
    # Gamma = (0, 0), X = (pi, 0), M = (pi, pi), Gamma = (0, 0)
    num_seg = 150
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)

    path_ka = np.vstack([k1, k2, k3])
    kx_a = path_ka[:, 0]
    ky_a = path_ka[:, 1]

    # Path parameter for x-axis
    path_x = np.linspace(0.0, 3.0, len(path_ka))
    tick_locs = [0.0, 1.0, 2.0, 3.0]
    tick_labels = [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$']

    # Initial band calculations
    energies_init = get_bands_along_path(a_init, kx_a, ky_a, d0=d0_val, t0=t0_init, mode=initial_mode)

    # Setup Figure and Axis
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.26)

    line1, = ax.plot(path_x, energies_init[:, 0], 'b-', lw=2.2, label=r'$E_1(\vec{k})$ (Lower band)')
    line2, = ax.plot(path_x, energies_init[:, 1], 'r-', lw=2.2, label=r'$E_2(\vec{k})$ (Upper band)')

    # On-site reference levels
    ref_eps1 = ax.axhline(0.0, color='blue', linestyle=':', alpha=0.4, label=r'$\epsilon_1 = 0$')
    ref_eps2 = ax.axhline(0.8 * t0_init, color='red', linestyle=':', alpha=0.4, label=r'$\epsilon_2 = 0.8 t_0$')

    for t in tick_locs:
        ax.axvline(t, color='gray', linestyle='--', alpha=0.6)

    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_ylabel(r'Energy ($E$)', fontsize=13)
    ax.set_xlabel(r'Wavevector $\vec{k}$ path', fontsize=12)
    title_text = ax.set_title(f'Square Lattice Band Structure [{initial_mode}] | $a = {a_init/d0_val:.2f} d_0$, $t_0 = {t0_init:.2f}$', fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)

    # Initial y-limits
    y_min = np.min(energies_init) - 0.2
    y_max = np.max(energies_init) + 0.2
    ax.set_ylim(y_min, y_max)

    # --- Widgets ---
    # Radio buttons for selecting model configuration
    ax_radio = plt.axes([0.06, 0.03, 0.20, 0.15])
    radio = RadioButtons(ax_radio, ('NN + NNN', 'NN only', 'No Overlap'), active=0)

    # Slider 1: Lattice parameter a (units of d0)
    ax_a = plt.axes([0.38, 0.11, 0.54, 0.03])
    a_slider = Slider(
        ax=ax_a,
        label=r'Lattice constant $a / d_0$',
        valmin=3.2,
        valmax=12.0,
        valinit=a_init / d0_val,
        valstep=0.05
    )

    # Slider 2: Energy parameter t_0
    ax_t0 = plt.axes([0.38, 0.04, 0.54, 0.03])
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
        mode_current = radio.value_selected

        new_energies = get_bands_along_path(a_current, kx_a, ky_a, d0=d0_val, t0=t0_current,
                                            mode=mode_current)

        line1.set_ydata(new_energies[:, 0])
        line2.set_ydata(new_energies[:, 1])

        ref_eps2.set_ydata([0.8 * t0_current, 0.8 * t0_current])
        title_text.set_text(f'Square Lattice Band Structure [{mode_current}] | $a = {a_current/d0_val:.2f} d_0$, $t_0 = {t0_current:.2f}$')

        cur_min = np.min(new_energies) - 0.15 * t0_current
        cur_max = np.max(new_energies) + 0.15 * t0_current
        ax.set_ylim(cur_min, cur_max)

        fig.canvas.draw_idle()

    a_slider.on_changed(update)
    t0_slider.on_changed(update)
    radio.on_clicked(update)

    plt.show()

if __name__ == '__main__':
    main()
