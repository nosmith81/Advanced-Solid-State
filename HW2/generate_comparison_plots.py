"""
Generate diagnostic plots comparing the user's formulation with the AI agent's suggestion
and visualizing the numerical stability and overlap matrix behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from compare_stability import run_path_comparison, solve_user_method, solve_agent_eigh, solve_symmetrized_eigh
from band_structure import calc_t_matrix, calc_eta_matrix

def generate_diagnostic_plots():
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------
    # Panel (a): a = 7.0 d0 - User vs Exact GenEig vs Symmetrized
    # -------------------------------------------------------------
    res7 = run_path_comparison(7.0)
    ax_a = axes[0, 0]
    ax_a.plot(res7['path_x'], res7['bands_user_real'][:, 0], 'b-', lw=2.5, label='User Formulation ($E_1$)')
    ax_a.plot(res7['path_x'], res7['bands_user_real'][:, 1], 'r-', lw=2.5, label='User Formulation ($E_2$)')
    ax_a.plot(res7['path_x'], res7['bands_geneig_real'][:, 0], 'k--', lw=1.5, label='Exact GenEig $Ac=\\lambda Bc$')
    ax_a.plot(res7['path_x'], res7['bands_geneig_real'][:, 1], 'k--', lw=1.5)
    ax_a.plot(res7['path_x'], res7['bands_symm'][:, 0], 'g:', lw=2.0, label='Symmetrized $H_{sym}$')
    ax_a.plot(res7['path_x'], res7['bands_symm'][:, 1], 'g:', lw=2.0)
    
    ax_a.set_title(r'(a) Bands at $a = 7.0 d_0$: User $M$ vs Exact GenEig vs Symmetrized', fontsize=11, fontweight='bold')
    ax_a.set_ylabel(r'Energy ($E / t_0$)', fontsize=11)
    ax_a.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax_a.set_xticklabels([r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'], fontsize=12)
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # Panel (b): a = 7.0 d0 - Failure of AI Agent's eigh(H, S)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.plot(res7['path_x'], res7['bands_user_real'][:, 0], 'k-', lw=2.0, label='True Bands (User $M$)')
    ax_b.plot(res7['path_x'], res7['bands_user_real'][:, 1], 'k-', lw=2.0)
    ax_b.plot(res7['path_x'], res7['bands_agent_lower'][:, 0], 'c--', lw=2.0, label=r'Agent `eigh` (lower triangle)')
    ax_b.plot(res7['path_x'], res7['bands_agent_lower'][:, 1], 'c--', lw=2.0)
    ax_b.plot(res7['path_x'], res7['bands_agent_upper'][:, 0], 'm-.', lw=2.0, label=r'Agent `eigh` (upper triangle)')
    ax_b.plot(res7['path_x'], res7['bands_agent_upper'][:, 1], 'm-.', lw=2.0)

    ax_b.set_title(r'(b) Failure of Agent `eigh`: $H = \bar{\epsilon}\bar{\eta} + \bar{t}$ is Non-Hermitian', fontsize=11, fontweight='bold')
    ax_b.set_ylabel(r'Energy ($E / t_0$)', fontsize=11)
    ax_b.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax_b.set_xticklabels([r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'], fontsize=12)
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # Panel (c): a = 5.0 d0 - User singularity and Cholesky Crash
    # -------------------------------------------------------------
    res5 = run_path_comparison(5.0)
    ax_c = axes[1, 0]
    ax_c.plot(res5['path_x'], res5['bands_user_real'][:, 0], 'b-', lw=2.0, label='User Lower Band')
    ax_c.plot(res5['path_x'], res5['bands_user_real'][:, 1], 'r-', lw=2.0, label='User Upper Band')
    ax_c.set_ylim(-50, 270)
    ax_c.set_title(r'(c) $a = 5.0 d_0$: Overlap Singularity (Agent `eigh` crashes 195/451 pts)', fontsize=11, fontweight='bold')
    ax_c.set_ylabel(r'Energy ($E / t_0$)', fontsize=11)
    ax_c.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax_c.set_xticklabels([r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'], fontsize=12)
    ax_c.grid(alpha=0.3)
    ax_c.legend(loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # Panel (d): Overlap Matrix Minimum Eigenvalue vs Lattice Constant a
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    a_scan = np.linspace(3.5, 11.0, 120)
    d0 = 1.0
    
    # Check minimum eigenvalue across BZ path for each a
    num_seg = 40
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)
    path_ka = np.vstack([k1, k2, k3])
    
    min_eig_nn_nnn = []
    min_eig_nn_only = []

    for a in a_scan:
        eta_nn = calc_eta_matrix(a, d0=d0)
        eta_nnn = calc_eta_matrix(np.sqrt(2.0)*a, d0=d0)
        
        m_full = float('inf')
        m_nn = float('inf')
        for kx, ky in path_ka:
            eps_0 = np.cos(kx) + np.cos(ky)
            eps_1 = 2.0 * np.cos(kx) * np.cos(ky)
            
            eta_full = np.eye(2) + 2.0 * eta_nn * eps_0 + 2.0 * eta_nnn * eps_1
            eta_nn_mat = np.eye(2) + 2.0 * eta_nn * eps_0
            
            ev_f = np.linalg.eigvalsh(eta_full)[0]
            ev_n = np.linalg.eigvalsh(eta_nn_mat)[0]
            if ev_f < m_full:
                m_full = ev_f
            if ev_n < m_nn:
                m_nn = ev_n
                
        min_eig_nn_nnn.append(m_full)
        min_eig_nn_only.append(m_nn)

    ax_d.plot(a_scan, min_eig_nn_nnn, 'b-', lw=2.2, label=r'NN + NNN: $\min_{\vec{k}} \lambda(\bar{\eta})$')
    ax_d.plot(a_scan, min_eig_nn_only, 'g--', lw=2.2, label=r'NN only: $\min_{\vec{k}} \lambda(\bar{\eta})$')
    ax_d.axhline(0.0, color='r', linestyle=':', lw=1.5, label='Zero (Singularity Threshold)')
    ax_d.axvline(5.42, color='b', linestyle='--', alpha=0.6, label=r'NN+NNN threshold ($a=5.42d_0$)')
    ax_d.axvline(7.20, color='g', linestyle='--', alpha=0.6, label=r'NN threshold ($a=7.20d_0$)')
    ax_d.axvline(5.0, color='red', linestyle='-', alpha=0.5, label=r'Initial $a=5.0d_0$ (Unphysical)')
    
    ax_d.set_title(r'(d) Overlap Matrix Positive-Definiteness vs Lattice Constant $a$', fontsize=11, fontweight='bold')
    ax_d.set_xlabel(r'Lattice Constant $a / d_0$', fontsize=11)
    ax_d.set_ylabel(r'Minimum Eigenvalue of $\bar{\eta}(\vec{k})$', fontsize=11)
    ax_d.grid(alpha=0.3)
    ax_d.legend(loc='lower right', fontsize=8.5)
    ax_d.set_ylim(-0.4, 0.8)

    plt.tight_layout()
    output_path = '/home/pseudonym/Documents/Code/Classes/Advanced_Solid_State_Stanescu_Fall2026/Advanced-Solid-State/HW2/stability_comparison_analysis.png'
    plt.savefig(output_path, dpi=200)
    print(f"Diagnostic plot saved successfully to {output_path}")

if __name__ == '__main__':
    generate_diagnostic_plots()
