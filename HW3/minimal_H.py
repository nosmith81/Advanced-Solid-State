#!/usr/bin/env python3
"""
minimal_H.py: Band structure and 2D Brillouin Zone heatmap calculation
for the minimal 4x4 ribbon Hamiltonian.

Source Reference:
  Main_Vault/Active/Periodic/Daily/2026/2026-11-03.md

Model Description:
  Square lattice with anisotropic hopping:
    - Lattice vectors: a_1 = a/sqrt(2) * (x + y),  a_2 = a/sqrt(2) * (x - y)
    - Nearest-neighbor hopping: t_1(k) = 2*t * [cos(k . a_1) + cos(k . a_2)]
                                       = 4*t * cos(k_x*a/sqrt(2)) * cos(k_y*a/sqrt(2))
    - Next-nearest-neighbor hopping along y: t_2(k) = (t/2) * cos(sqrt(2)*a*k_y)
  
  Minimal 4x4 Hamiltonian block matrix:
    H(k) = [[ H_cell,      T ],
            [ T^dagger, H_cell ]]
    where:
      H_cell = [[ epsilon, t_1     ],
                [ t_1,     epsilon ]]
      T      = [[ t_2,     0       ],
                [ t_1,     t_2     ]]
"""

import os
import sys
import numpy as np
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# 1. Physical Parameters & Brillouin Zone Definition
# =====================================================================
t = 1.0        # Hopping energy scale
epsilon = 0.0  # On-site atomic energy
a = 1.0        # Lattice constant

# BZ boundary: k in [-k_max, +k_max] along both k_x and k_y
k_max = np.pi / (np.sqrt(2.0) * a)

print("=" * 60)
print("Minimal 4x4 Hamiltonian Band Structure Calculation")
print("=" * 60)
print(f"Parameters: t = {t}, epsilon = {epsilon}, a = {a}")
print(f"Brillouin Zone boundary: [-pi/(sqrt(2)*a), +pi/(sqrt(2)*a)]")
print(f"k_max = {k_max:.5f} (1/length)\n")



def calc_t1(k):
    kx, ky = k
    return -t * (1 + np.exp(-np.sqrt(2)*a*kx))

def calc_t2(k):
    kx, ky = k
    return -0.25*t

def build_cell_and_hopping(t1, t2, eps=epsilon):
    """
    Constructs Hcell and T tensors given t1 and t2 (scalars or momentum arrays).
      Hcell = [[ eps, t1  ],
               [ t1,  eps ]]
      T     = [[ t2,  0   ],
               [ t1,  t2  ]]
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t1, t2 = np.broadcast_arrays(t1, t2)
    eps_arr = np.full_like(t1, eps)
    zeros_arr = np.zeros_like(t1)

    Hcell = np.stack([
        np.stack([eps_arr, t1], axis=-1),
        np.stack([t1, eps_arr], axis=-1)
    ], axis=-2)

    T = np.stack([
        np.stack([t2, zeros_arr], axis=-1),
        np.stack([t1, t2], axis=-1)
    ], axis=-2)
    return Hcell, T

def construct_hamiltonian_NxN(N, Hcell, T, config='Z'):
    """
    returns the Hamiltonian matrix as a function of k for a ribbon of width N unit cells.
    Operates seamlessly on both single k-point matrices (shape (2, 2))
    and full momentum-space tensors (shape (..., 2, 2)).

    Parameters:
      N: number of 2-site unit cells coupled by hopping matrix T.
      Hcell: 2x2 intra-cell Hamiltonian (or (..., 2, 2) tensor).
      T: 2x2 inter-cell hopping matrix (or (..., 2, 2) tensor).
      config: 'Z' (default) or 'W'.
        - 'Z': 2N sites (even). Starts on Line 1 (bottom) and ends on Line 2 (top).
          Lacks reflection symmetry about the center horizontal line of the ribbon.
          Matrix dimension: 2N x 2N.
        - 'W': 2N + 1 sites (odd). Appends an extra site (site 2N) on Line 1 with its exact
          tight-binding hoppings (t1 to site 2N-1, t2 to site 2N-2, on-site epsilon).
          Starts on Line 1 and ends on Line 1, establishing exact reflection symmetry
          about the center horizontal line of the crystal.
          Matrix dimension: (2N + 1) x (2N + 1).
    """
    Hcell = np.asarray(Hcell)
    T = np.asarray(T)
    Tdag = np.swapaxes(np.conjugate(T), -1, -2)

    if N < 2:
        if N == 1:
            H = Hcell
        else:
            raise ValueError("N should be greater than or equal to 1!")
    else:
        # Initial 2-block Hamiltonian (minimal 4x4 block)
        top_row = np.concatenate([Hcell, T], axis=-1)
        bottom_row = np.concatenate([Tdag, Hcell], axis=-1)
        H = np.concatenate([top_row, bottom_row], axis=-2)

        prefix = Hcell.shape[:-2]
        for n in range(N - 2):
            curr_dim = H.shape[-2]
            zeros_tr = np.zeros((*prefix, curr_dim - 2, 2), dtype=Hcell.dtype)
            top_right = np.concatenate([zeros_tr, T], axis=-2)

            zeros_bl = np.zeros((*prefix, 2, curr_dim - 2), dtype=Hcell.dtype)
            bottom_left = np.concatenate([zeros_bl, Tdag], axis=-1)

            top = np.concatenate([H, top_right], axis=-1)
            bottom = np.concatenate([bottom_left, Hcell], axis=-1)
            H = np.concatenate([top, bottom], axis=-2)

    config_str = str(config).upper().strip()
    if config_str == 'W':
        # Append extra site (site 2N) on Line 1 to switch from Z to W configuration.
        # This restores reflection symmetry about the center horizontal line of the ribbon:
        # - Site 2N is on Line 1, so the chain starts on Line 1 (site 0) and ends on Line 1 (site 2N).
        # - Diagonal on-site energy: epsilon = Hcell[..., 0, 0]
        # - Nearest-neighbor hopping (chain distance 1 to site 2N-1 on Line 2): t1 = Hcell[..., 0, 1]
        # - Next-nearest-neighbor hopping (chain distance 2 to site 2N-2 on Line 1): t2 = T[..., 0, 0]
        prefix = Hcell.shape[:-2]
        dim_W = 2 * N + 1
        H_W = np.zeros((*prefix, dim_W, dim_W), dtype=Hcell.dtype)
        H_W[..., :2 * N, :2 * N] = H
        H_W[..., 2 * N, 2 * N] = Hcell[..., 0, 0]
        H_W[..., 2 * N, 2 * N - 1] = Hcell[..., 0, 1]
        H_W[..., 2 * N - 1, 2 * N] = np.conjugate(Hcell[..., 0, 1])
        H_W[..., 2 * N, 2 * N - 2] = T[..., 0, 0]
        H_W[..., 2 * N - 2, 2 * N] = np.conjugate(T[..., 0, 0])
        return H_W
    elif config_str == 'Z':
        return H
    else:
        raise ValueError(f"Unknown configuration '{config}'. Expected 'Z' or 'W'.")






def compute_hamiltonian_4x4(kx, ky):
    """
    Explicit 4x4 (2 x W, W=2) minimal Hamiltonian matrix at wavevector (kx, ky).
    Direct benchmark matrix from the analytical block formulation:
      H = [[ Hcell, T ],
           [ Tdag,  Hcell ]]
    """
    t1 = calc_t1((kx, ky))
    t2 = calc_t2((kx, ky))
    H = np.array([
        [epsilon, t1,      t2,      0.0    ],
        [t1,      epsilon, t1,      t2     ],
        [t2,      t1,      epsilon, t1     ],
        [0.0,     t2,      t1,      epsilon]
    ], dtype=float)
    return H

# Alias for 2xW formulation (W=2)
compute_hamiltonian_2xw = compute_hamiltonian_4x4


def compute_transverse_wavefunctions(N, kx=0.0, ky=0.0, config='Z'):
    """
    Computes transverse wave function profiles in the y-direction across the ribbon
    for all eigenstates GIVEN a specific momentum eigenstate (kx, ky).

    The full unitary matrix U diagonalizes the ribbon Hamiltonian H(k) and satisfies
    U^dagger @ U = I to machine precision.

    Conditioning is performed ONLY on the k-momentum eigenstate. The vertical lines
    are NOT conditioned, so each line is NOT individually renormalized to 1.
    Instead, the probability measure over the entire transverse coordinate space
    satisfies:
        sum_m p_even[m, n] + sum_m p_odd[m, n] = 1.0

    Configurations:
      - 'Z': 2N sites (N sites on Line 1, N sites on Line 2).
      - 'W': 2N + 1 sites (N + 1 sites on Line 1, N sites on Line 2),
             exhibiting exact reflection symmetry about the center horizontal line of the crystal.

    Returns:
      evals: sorted eigenvalues, shape (dim,)
      evecs: unitary matrix of eigenvectors U, shape (dim, dim)
      prob_density: |evecs|^2, shape (dim, dim)
      p_even: prob_density[0::2, :], Line 1 (even sites, length N for Z, N+1 for W)
      p_odd:  prob_density[1::2, :], Line 2 (odd sites, length N)
    """
    t1 = calc_t1((kx, ky))
    t2 = calc_t2((kx, ky))
    Hcell, T = build_cell_and_hopping(t1, t2, eps=epsilon)
    H = construct_hamiltonian_NxN(N, Hcell, T, config=config)
    evals, evecs = np.linalg.eigh(H)
    dim = H.shape[-1]

    # Unitary matrix check: U^\dagger @ U = I and U @ U^\dagger = I
    I_expected = np.eye(dim)
    U_dag_U = evecs.conj().T @ evecs
    U_U_dag = evecs @ evecs.conj().T
    np.testing.assert_allclose(U_dag_U, I_expected, atol=1e-14, rtol=1e-14,
                              err_msg="Eigenvector matrix U is not unitary: U^dagger @ U != I")
    np.testing.assert_allclose(U_U_dag, I_expected, atol=1e-14, rtol=1e-14,
                              err_msg="Eigenvector matrix U is not unitary: U @ U^dagger != I")

    # Mod-square of normalized eigenvectors: |U_{j, n}|^2
    prob_density = np.abs(evecs) ** 2

    # Assert exact total normalization across the transverse space: sum_j |U_{j, n}|^2 = 1.0
    np.testing.assert_allclose(np.sum(prob_density, axis=0), 1.0, atol=1e-14, rtol=1e-14)

    # Transverse profiles along Vertical Line 1 (even sites) and Vertical Line 2 (odd sites)
    p_even = prob_density[0::2, :]  # shape (N or N+1, dim): Line 1 (even sites)
    p_odd = prob_density[1::2, :]   # shape (N, dim): Line 2 (odd sites)

    # Total transverse probability along y sums to 1.0:
    # sum_m p_even + sum_m p_odd = 1.0
    total_transverse = np.sum(p_even, axis=0) + np.sum(p_odd, axis=0)
    np.testing.assert_allclose(total_transverse, 1.0, atol=1e-14, rtol=1e-14,
                              err_msg="Total transverse sum != 1.0!")

    return evals, evecs, prob_density, p_even, p_odd


def test_N2_parity():
    """
    Asserts that the iterative construct_hamiltonian_NxN with N=2 produces
    eigenvalues identical to machine precision compared to the explicit 4x4 (2xW) function.
    """
    print("Running N=2 parity test against explicit 4x4 (2xW) function...")
    test_kx = np.linspace(-k_max, k_max, 25)
    test_ky = np.linspace(-k_max, k_max, 25)

    max_H_err = 0.0
    max_eig_err = 0.0

    for kx_val in test_kx:
        for ky_val in test_ky:
            H_explicit = compute_hamiltonian_4x4(kx_val, ky_val)

            t1_val = calc_t1((kx_val, ky_val))
            t2_val = calc_t2((kx_val, ky_val))
            Hcell, T = build_cell_and_hopping(t1_val, t2_val, eps=epsilon)
            H_iter = construct_hamiltonian_NxN(2, Hcell, T)

            diff_H = np.max(np.abs(H_iter - H_explicit))
            if diff_H > max_H_err:
                max_H_err = diff_H

            e_explicit = np.linalg.eigvalsh(H_explicit)
            e_iter = np.linalg.eigvalsh(H_iter)
            diff_eig = np.max(np.abs(e_iter - e_explicit))
            if diff_eig > max_eig_err:
                max_eig_err = diff_eig

            np.testing.assert_allclose(H_iter, H_explicit, atol=1e-15, rtol=1e-15,
                                      err_msg=f"Matrix mismatch at ({kx_val}, {ky_val})")
            np.testing.assert_allclose(e_iter, e_explicit, atol=1e-14, rtol=1e-14,
                                      err_msg=f"Eigenvalue mismatch at ({kx_val}, {ky_val})")

    print(f"  Max Hamiltonian matrix difference: {max_H_err:.2e}")
    print(f"  Max eigenvalue difference:         {max_eig_err:.2e}")
    print("  [PASS] All N=2 eigenvalues match the 2xW function to machine precision!\n")
    return True


def run_analysis():
    # Run verification test
    test_N2_parity()

    # =====================================================================
    # 3. 1D Momentum Space (kx) Dispersions
    # =====================================================================
    # Ribbons are finite along y (no ky quantum number) and periodic along x (kx).
    N_kx = 300
    kx_path = np.linspace(-k_max, k_max, N_kx)
    ky_dummy = np.zeros_like(kx_path)

    # Hopping evaluations along kx
    T1_path = calc_t1((kx_path, ky_dummy))
    T2_path = calc_t2((kx_path, ky_dummy))
    Hcell_path, T_path = build_cell_and_hopping(T1_path, T2_path, eps=epsilon)

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # =====================================================================
    # 4. Loop over N: N increased by 5 up to 20
    # =====================================================================
    N_values = [2, 5, 10, 15, 20]
    comparison_data = {}

    for N in N_values:
        num_bands = 2 * N
        print(f"\n{'=' * 60}")
        print(f"Computing Band Structure for N = {N} ({num_bands} bands)")
        print(f"{'=' * 60}")

        # 1D Ribbon Diagonalization along kx
        H_path = construct_hamiltonian_NxN(N, Hcell_path, T_path)
        bands_path = np.linalg.eigvalsh(H_path)
        comparison_data[N] = bands_path

        # Print summary metrics
        e_min = np.min(bands_path)
        e_max = np.max(bands_path)
        gap_zero = np.min(bands_path[:, N] - bands_path[:, N - 1])
        print(f"  Energy span: [{e_min:.3f}, {e_max:.3f}] t")
        print(f"  Min bandgap around E=0: {gap_zero:.4f} t")

        # Specific kx points summary: -k_max, 0 (Gamma), +k_max
        test_kx_pts = [
            (-k_max, r'$-k_{\mathrm{max}}$'),
            (0.0, r'$\Gamma$ ($k_x=0$)'),
            (k_max, r'$+k_{\mathrm{max}}$')
        ]
        print("  Eigenvalues at representative kx points:")
        for kx_val, pt_name in test_kx_pts:
            t1_pt = calc_t1((kx_val, 0.0))
            t2_pt = calc_t2((kx_val, 0.0))
            Hcell_pt, T_pt = build_cell_and_hopping(t1_pt, t2_pt, eps=epsilon)
            H_pt = construct_hamiltonian_NxN(N, Hcell_pt, T_pt)
            e_pt = np.linalg.eigvalsh(H_pt)
            print(f"    {pt_name:>15} (kx={kx_val:.3f}): "
                  f"E_min = {e_pt[0]:6.3f}, E_mid = [{e_pt[N-1]:6.3f}, {e_pt[N]:6.3f}], E_max = {e_pt[-1]:6.3f}")

        # Visualization for each N along kx
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.cm.tab10 if num_bands <= 10 else plt.cm.viridis
        for b in range(num_bands):
            c = cmap(b % 10 if num_bands <= 10 else b / num_bands)
            ax.plot(kx_path, bands_path[:, b], lw=1.4, color=c, alpha=0.85)

        ax.axvline(0.0, color='gray', linestyle=':', lw=1.0, alpha=0.8)
        ax.set_xlim([-k_max, k_max])
        ax.set_xticks([-k_max, -k_max / 2, 0.0, k_max / 2, k_max])
        ax.set_xticklabels([r'$-k_{\mathrm{max}}$', r'$-k_{\mathrm{max}}/2$', r'$0$', r'$k_{\mathrm{max}}/2$', r'$k_{\mathrm{max}}$'], fontsize=11)
        ax.set_xlabel(r'Wavevector $k_x$', fontsize=12)
        ax.set_ylabel(r'Energy $E / t$', fontsize=12)
        ax.set_title(f'Ribbon Dispersion along $k_x$ (N = {N}, {num_bands} bands)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()

        out_file = os.path.join(output_dir, f"minimal_H_bands_N{N}.png")
        plt.savefig(out_file, dpi=200)
        print(f"  Figure saved to: {out_file}")

        if N == 2:
            out_base = os.path.join(output_dir, "minimal_H_bands.png")
            plt.savefig(out_base, dpi=200)

        if "--show" in sys.argv:
            plt.show()
        plt.close(fig)

    # =====================================================================
    # 5. Multi-N Band Structure Comparison Figure
    # =====================================================================
    comp_N = [n for n in N_values if n in [5, 10, 15, 20]]
    fig_comp, axes = plt.subplots(1, len(comp_N), figsize=(4.2 * len(comp_N), 5.5), sharey=True)

    for idx, N in enumerate(comp_N):
        ax = axes[idx]
        b_path = comparison_data[N]
        num_b = 2 * N
        cmap = plt.cm.viridis
        for b in range(num_b):
            ax.plot(kx_path, b_path[:, b], lw=1.2, color=cmap(b / num_b), alpha=0.8)

        ax.axvline(0.0, color='gray', linestyle=':', lw=1.0, alpha=0.7)
        ax.set_xlim([-k_max, k_max])
        ax.set_xticks([-k_max, 0.0, k_max])
        ax.set_xticklabels([r'$-k_{\mathrm{max}}$', r'$0$', r'$k_{\mathrm{max}}$'], fontsize=11)
        ax.set_title(f'N = {N} ({num_b} bands)', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Energy $E / t$', fontsize=11)
        ax.set_xlabel(r'Wavevector $k_x$', fontsize=11)

    fig_comp.suptitle(r'Ribbon Dispersion Evolution with Width N along $k_x$ (N = 5, 10, 15, 20)', fontsize=13, y=1.02)
    plt.tight_layout()
    comp_file = os.path.join(output_dir, "minimal_H_bands_comparison.png")
    plt.savefig(comp_file, dpi=200, bbox_inches='tight')
    print(f"\nMulti-N comparison figure saved to: {comp_file}")

    if "--show" in sys.argv:
        plt.show()
    plt.close(fig_comp)

    # =====================================================================
    # 6. Transverse Wave Function Profiles in y-direction (3 Figures)
    # =====================================================================
    # In each unit cell along y, even indices correspond to Vertical Line 1 (Sublattice A),
    # and odd indices correspond to Vertical Line 2 (Sublattice B).
    N_trans = 100  # Ribbon width: N_trans unit cells along y (200 sites total)
    num_sites = 2 * N_trans
    y_cell_idx = np.arange(N_trans)  # 0 to N_trans - 1

    print(f"\n{'=' * 60}")
    print(f"Computing Transverse Wave Function Profiles for N = {N_trans} ({num_sites} sites)")
    print(f"  Vertical Line 1 (Sublattice A): Even indices (0, 2, ..., {2*N_trans-2})")
    print(f"  Vertical Line 2 (Sublattice B): Odd indices  (1, 3, ..., {2*N_trans-1})")
    print(f"  Generating 3 figures: Ground State, Middle Energy State, Highest Energy State")
    print(f"{'=' * 60}")

    # Representative kx Points along the 1D BZ
    kx_points = [
        (r'$-k_{\mathrm{max}}$', -k_max, r'-\pi/(\sqrt{2}a)'),
        (r'$-k_{\mathrm{max}}/2$', -k_max / 2.0, r'-\pi/(2\sqrt{2}a)'),
        (r'$\Gamma$ ($k_x=0$)', 0.0, '0'),
        (r'$+k_{\mathrm{max}}$', k_max, r'+\pi/(\sqrt{2}a)'),
    ]

    # Precompute wavefunctions at representative kx points
    kx_data = {}
    for pt_name, kx_val, kx_str in kx_points:
        evals_pt, evecs_pt, prob_pt, pe_pt, po_pt = compute_transverse_wavefunctions(
            N_trans, kx=kx_val, ky=0.0
        )
        kx_data[pt_name] = {
            'evals': evals_pt,
            'p_even': pe_pt,
            'p_odd': po_pt,
            'kx_str': kx_str,
        }

    # 3 Target states to plot across 3 separate figures
    state_specs = [
        ("Ground State", 0, "ground", "Band 1"),
        ("Middle Energy State", N_trans - 1, "mid", f"Band {N_trans}"),
        ("Highest Energy State", 2 * N_trans - 1, "highest", f"Band {2 * N_trans}")
    ]

    generated_figures = []

    for state_title, band_idx, file_suffix, band_label in state_specs:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(
            f'Transverse Wave Function Profiles — {state_title} ({band_label} of {num_sites}, N = {N_trans})',
            fontsize=14, fontweight='bold', y=0.98
        )

        print(f"\n  === {state_title} ({band_label}) ===")
        for s_idx, (pt_name, kx_val, kx_str) in enumerate(kx_points):
            row = s_idx // 2
            col = s_idx % 2
            ax = axes[row, col]

            data = kx_data[pt_name]
            E_val = data['evals'][band_idx]
            pe = data['p_even'][:, band_idx]
            po = data['p_odd'][:, band_idx]
            w1 = np.sum(pe)
            w2 = np.sum(po)

            ax.plot(y_cell_idx, pe, 'b.-', lw=1.2, ms=2.5, label=f'Line 1 (Even, $\\sum={w1:.4f}$)')
            ax.plot(y_cell_idx, po, 'r.--', lw=1.2, ms=2.5, label=f'Line 2 (Odd, $\\sum={w2:.4f}$)')

            ax.set_title(fr'{pt_name}: $k_x={kx_str}$ — $E = {E_val:.3f}t$', fontsize=11)
            ax.set_xlabel('Transverse Unit Cell Index $m$ along $y$ (0 to $N-1$)', fontsize=10)
            ax.set_ylabel(r'Probability Density $|\psi(y_m)|^2$', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(fontsize=9, loc='best')

            print(f"    {pt_name:>18}: E = {E_val:7.3f}t | Line 1 sum = {w1:.4f}, Line 2 sum = {w2:.4f}, total = {w1+w2:.4f}")

        plt.tight_layout()
        out_fig_file = os.path.join(output_dir, f"minimal_H_transverse_{file_suffix}.png")
        plt.savefig(out_fig_file, dpi=200, bbox_inches='tight')
        out_fig_n100 = os.path.join(output_dir, f"minimal_H_transverse_{file_suffix}_N{N_trans}.png")
        plt.savefig(out_fig_n100, dpi=200, bbox_inches='tight')
        print(f"  Saved figure: {out_fig_file}")
        generated_figures.append(out_fig_file)

        if "--show" in sys.argv:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    run_analysis()

