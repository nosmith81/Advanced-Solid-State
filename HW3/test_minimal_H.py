#!/usr/bin/env python3
"""
test_minimal_H.py: Machine-precision parity test for N=2 Hamiltonian.
Compares the iterative tensor constructor `construct_hamiltonian_NxN(2, Hcell, T)`
against the analytical 4x4 (2xW with W=2) benchmark function `compute_hamiltonian_4x4`.
"""

import os
import sys
import numpy as np

# Ensure HW3 module directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minimal_H import (
    t, epsilon, a, k_max,
    calc_t1, calc_t2,
    build_cell_and_hopping,
    construct_hamiltonian_NxN,
    compute_hamiltonian_4x4,
    compute_hamiltonian_2xw,
    compute_transverse_wavefunctions
)


def test_n2_high_symmetry_points():
    """
    Asserts machine-precision equality of eigenvalues at high-symmetry points:
    Gamma (0, 0), X (k_max, 0), M (k_max, k_max), Y (0, k_max).
    """
    points = {
        "Gamma": (0.0, 0.0),
        "X":     (k_max, 0.0),
        "M":     (k_max, k_max),
        "Y":     (0.0, k_max),
    }

    print("\n--- Testing High-Symmetry Points Parity (N = 2 vs 2xW) ---")
    for name, (kx, ky) in points.items():
        # 1. Benchmark 2xW (4x4) matrix
        H_2xw = compute_hamiltonian_2xw(kx, ky)
        evals_2xw = np.linalg.eigvalsh(H_2xw)

        # 2. Iterative N=2 matrix
        t1 = calc_t1((kx, ky))
        t2 = calc_t2((kx, ky))
        Hcell, T = build_cell_and_hopping(t1, t2, eps=epsilon)
        H_N2 = construct_hamiltonian_NxN(2, Hcell, T)
        evals_N2 = np.linalg.eigvalsh(H_N2)

        # Assertions
        matrix_diff = np.max(np.abs(H_N2 - H_2xw))
        eval_diff = np.max(np.abs(evals_N2 - evals_2xw))

        print(f"  Point {name:>5} ({kx:.3f}, {ky:.3f}): "
              f"max |Delta H| = {matrix_diff:.2e}, max |Delta E| = {eval_diff:.2e}")

        np.testing.assert_allclose(
            H_N2, H_2xw, atol=1e-15, rtol=1e-15,
            err_msg=f"Matrix elements differ at {name}!"
        )
        np.testing.assert_allclose(
            evals_N2, evals_2xw, atol=1e-14, rtol=1e-14,
            err_msg=f"Eigenvalues differ at {name}!"
        )

    print("  [PASS] All high-symmetry points match to machine precision.")


def test_n2_full_momentum_grid():
    """
    Asserts machine-precision equality across a dense 2D Brillouin Zone grid
    by evaluating the full (N_grid, N_grid, 4, 4) momentum tensor simultaneously.
    """
    print("\n--- Testing Full 2D Brillouin Zone Grid Parity (Dense Tensor) ---")
    N_grid = 50
    kx = np.linspace(-k_max, k_max, N_grid)
    ky = np.linspace(-k_max, k_max, N_grid)
    KX, KY = np.meshgrid(kx, ky)

    # 1. Full tensor via construct_hamiltonian_NxN
    T1_grid = calc_t1((KX, KY))
    T2_grid = calc_t2((KX, KY))
    Hcell_grid, T_grid = build_cell_and_hopping(T1_grid, T2_grid, eps=epsilon)
    H_tensor_N2 = construct_hamiltonian_NxN(2, Hcell_grid, T_grid)
    evals_tensor_N2 = np.linalg.eigvalsh(H_tensor_N2)

    # 2. Benchmark evaluation point-by-point via 2xW function
    evals_benchmark = np.zeros((N_grid, N_grid, 4))
    H_benchmark = np.zeros((N_grid, N_grid, 4, 4))
    for i in range(N_grid):
        for j in range(N_grid):
            H_ij = compute_hamiltonian_2xw(KX[i, j], KY[i, j])
            H_benchmark[i, j] = H_ij
            evals_benchmark[i, j] = np.linalg.eigvalsh(H_ij)

    max_H_diff = np.max(np.abs(H_tensor_N2 - H_benchmark))
    max_eval_diff = np.max(np.abs(evals_tensor_N2 - evals_benchmark))

    print(f"  Grid size: {N_grid}x{N_grid} ({N_grid * N_grid} k-points)")
    print(f"  Max |Delta H(k)| across full BZ tensor: {max_H_diff:.2e}")
    print(f"  Max |Delta E(k)| across full BZ tensor: {max_eval_diff:.2e}")

    np.testing.assert_allclose(
        H_tensor_N2, H_benchmark, atol=1e-15, rtol=1e-15,
        err_msg="Full momentum-space Hamiltonian tensor mismatch!"
    )
    np.testing.assert_allclose(
        evals_tensor_N2, evals_benchmark, atol=1e-14, rtol=1e-14,
        err_msg="Momentum-space eigenvalues mismatch across BZ!"
    )
    print("  [PASS] Full 2D BZ momentum tensor matches to machine precision.")


def test_n2_random_k_points():
    """
    Asserts machine-precision equality across 200 random wavevectors in k-space.
    """
    print("\n--- Testing Random Wavevector Parity ---")
    np.random.seed(12345)
    random_k = np.random.uniform(-2 * k_max, 2 * k_max, size=(200, 2))

    max_eval_diff = 0.0
    for kx, ky in random_k:
        H_2xw = compute_hamiltonian_2xw(kx, ky)
        e_2xw = np.linalg.eigvalsh(H_2xw)

        t1 = calc_t1((kx, ky))
        t2 = calc_t2((kx, ky))
        Hcell, T = build_cell_and_hopping(t1, t2, eps=epsilon)
        H_N2 = construct_hamiltonian_NxN(2, Hcell, T)
        e_N2 = np.linalg.eigvalsh(H_N2)

        diff = np.max(np.abs(e_N2 - e_2xw))
        if diff > max_eval_diff:
            max_eval_diff = diff

        np.testing.assert_allclose(
            e_N2, e_2xw, atol=1e-14, rtol=1e-14,
            err_msg=f"Eigenvalues differ at random k=({kx}, {ky})!"
        )

    print(f"  Sampled 200 random k-points: max |Delta E| = {max_eval_diff:.2e}")
    print("  [PASS] Random k-points test passed to machine precision.")


def test_transverse_profile_normalization():
    """
    Asserts that:
      1. The eigenvector matrix U is strictly unitary: U^dagger @ U = I and U @ U^dagger = I
         to machine precision (< 1e-14).
      2. The full probability density across all 2N sites sums to 1.0 for every eigenstate:
         sum_{j=0}^{2N-1} |U_{j, n}|^2 = 1.0 (< 1e-14).
      3. The sum of the transverse profiles over both vertical lines equals 1.0 for each eigenstate:
         sum_{m=0}^{N-1} (p_even[m, n] + p_odd[m, n]) = 1.0 (< 1e-14).
      4. Spatial inversion symmetry (mapping Line 1 from bottom to Line 2 from top)
         holds to machine precision (< 1e-13) at Gamma.
    """
    print("\n--- Testing Unitary Eigenvector Normalization (U^dagger @ U = I) & Inversion Symmetry ---")
    test_widths = [2, 5, 10, 20, 100]
    test_points = [
        ("Gamma", 0.0, 0.0),
        ("X", k_max, 0.0),
        ("M", k_max, k_max),
        ("Y", 0.0, k_max),
    ]

    for N in test_widths:
        I_N = np.eye(2 * N)
        for pt_name, kx, ky in test_points:
            evals, U, prob_density, p_even, p_odd = compute_transverse_wavefunctions(
                N, kx=kx, ky=ky
            )

            # 1. Unitarity of eigenvector matrix: U^dagger @ U = I and U @ U^dagger = I
            U_dag_U = U.conj().T @ U
            U_U_dag = U @ U.conj().T
            np.testing.assert_allclose(U_dag_U, I_N, atol=1e-14, rtol=1e-14,
                                      err_msg=f"U^dagger @ U != I at N={N}, {pt_name}")
            np.testing.assert_allclose(U_U_dag, I_N, atol=1e-14, rtol=1e-14,
                                      err_msg=f"U @ U^dagger != I at N={N}, {pt_name}")

            # 2. Total eigenstate probability normalization across all 2N sites
            total_prob = np.sum(prob_density, axis=0)
            np.testing.assert_allclose(total_prob, 1.0, atol=1e-14, rtol=1e-14,
                                      err_msg=f"Total norm violation at N={N}, {pt_name}")

            # 3. Transverse probability conservation across both vertical lines: sum_m (p_even + p_odd) = 1.0
            unit_cell_sums = np.sum(p_even + p_odd, axis=0)
            np.testing.assert_allclose(unit_cell_sums, 1.0, atol=1e-14, rtol=1e-14,
                                      err_msg=f"Unit-cell probability sum != 1.0 at N={N}, {pt_name}")

            # 4. Verification that states are conditioned ONLY on k:
            # Individual vertical lines carry physical weights w_1, w_2 in [0, 1] with w_1 + w_2 = 1.0.
            # (Lines are NOT conditioned individually, so sum_m p_even != 1 in general).
            w_even = np.sum(p_even, axis=0)
            w_odd = np.sum(p_odd, axis=0)
            np.testing.assert_allclose(w_even + w_odd, 1.0, atol=1e-14, rtol=1e-14)
            assert np.all(w_even >= -1e-14) and np.all(w_even <= 1.0 + 1e-14)
            assert np.all(w_odd >= -1e-14) and np.all(w_odd <= 1.0 + 1e-14)

            # 5. Inversion symmetry at Gamma: Line 1 from bottom equals Line 2 from top
            if pt_name == "Gamma":
                inv_diff = np.max(np.abs(p_even - p_odd[::-1, :]))
                np.testing.assert_allclose(inv_diff, 0.0, atol=1e-13,
                                          err_msg=f"Inversion symmetry violated at Gamma for N={N}")

    print("  [PASS] Eigenvector matrix U strictly unitary (U^dagger @ U = I, U @ U^dagger = I) to < 1e-14.")
    print("  [PASS] Total eigenstate normalization sum_j |U_{j,n}|^2 = 1.0 verified across all N and k.")
    print("  [PASS] Total transverse unit-cell sum_m (p_even + p_odd) = 1.0 verified.")
    print("  [PASS] Conservation across lines w_even + w_odd = 1.0 verified across all N and k.")
    print("  [PASS] Inversion symmetry between Line 1 (bottom) and Line 2 (top) verified to < 1e-13.")


def test_w_configuration_reflection_symmetry():
    """
    Asserts exact reflection symmetry for the W-configuration about the center horizontal line of the crystal:
      1. Dimension is 2N + 1 (odd number of sites).
      2. Starts on Line 1 (site 0) and ends on Line 1 (site 2N).
      3. Under reflection across the center horizontal line, site j maps to 2N - j:
         - Even sites (Line 1) map to even sites: 2N - (2m) = 2(N - m) (Line 1).
         - Odd sites (Line 2) map to odd sites:   2N - (2m + 1) = 2(N - 1 - m) + 1 (Line 2).
         Both vertical lines are preserved under reflection!
      4. Commutator [H_W, J] = 0 holds to machine precision (< 1e-14) across all k-points.
      5. Eigenvectors have definite reflection parity: J |psi_n> = +/- |psi_n> (|lambda_n| = 1.0).
      6. Transverse probability profiles on both vertical lines are symmetric about the crystal center:
         - Line 1: p_even[m] = p_even[N - m] to < 1e-14
         - Line 2: p_odd[m]  = p_odd[N - 1 - m] to < 1e-14
      7. In contrast, Z-configuration explicitly lacks horizontal reflection symmetry: [H_Z, R_y] != 0.
      8. Full momentum tensor broadcasting works for shape (..., 2N+1, 2N+1).
    """
    print("\n--- Testing W-Configuration Reflection Symmetry about Center Horizontal Line ---")
    test_widths = [2, 3, 5, 10, 20]
    test_points = [
        ("Gamma", 0.0, 0.0),
        ("X", k_max, 0.0),
        ("M", k_max, k_max),
        ("Y", 0.0, k_max),
    ]

    for N in test_widths:
        dim_W = 2 * N + 1
        J = np.fliplr(np.eye(dim_W))  # J_{j, k} = delta_{j, 2N - k}

        for pt_name, kx, ky in test_points:
            t1 = calc_t1((kx, ky))
            t2 = calc_t2((kx, ky))
            Hcell, T = build_cell_and_hopping(t1, t2, eps=epsilon)

            # 1. Construct W Hamiltonian
            H_W = construct_hamiltonian_NxN(N, Hcell, T, config='W')
            assert H_W.shape == (dim_W, dim_W), f"Expected shape ({dim_W}, {dim_W}), got {H_W.shape}"

            # 2. Reflection commutator [H_W, J] = 0
            comm = J @ H_W - H_W @ J
            max_comm = np.max(np.abs(comm))
            np.testing.assert_allclose(max_comm, 0.0, atol=1e-14,
                                      err_msg=f"W-configuration [H_W, J] != 0 at N={N}, {pt_name}!")

            # 3. Diagonalization and wave functions
            evals, U, prob_density, p_even, p_odd = compute_transverse_wavefunctions(
                N, kx=kx, ky=ky, config='W'
            )
            assert p_even.shape == (N + 1, dim_W)
            assert p_odd.shape == (N, dim_W)

            # 4. Total probability normalization
            np.testing.assert_allclose(np.sum(prob_density, axis=0), 1.0, atol=1e-14)
            np.testing.assert_allclose(np.sum(p_even, axis=0) + np.sum(p_odd, axis=0), 1.0, atol=1e-14)

            # 5. Parity eigenvalues: J |psi> = +/- |psi>
            for band in range(dim_W):
                psi = U[:, band]
                J_psi = J @ psi
                parity = np.real(psi.conj().T @ J_psi)
                np.testing.assert_allclose(np.abs(parity), 1.0, atol=1e-13,
                                          err_msg=f"Band {band} does not have definite parity!")

            # 6. Transverse profile reflection symmetry on Line 1 and Line 2
            diff_line1 = np.max(np.abs(p_even - p_even[::-1, :]))
            diff_line2 = np.max(np.abs(p_odd - p_odd[::-1, :]))
            np.testing.assert_allclose(diff_line1, 0.0, atol=1e-14,
                                      err_msg=f"Line 1 reflection asymmetry at N={N}, {pt_name}!")
            np.testing.assert_allclose(diff_line2, 0.0, atol=1e-14,
                                      err_msg=f"Line 2 reflection asymmetry at N={N}, {pt_name}!")

            # 7. Contrast with Z-configuration: lacks reflection symmetry
            H_Z = construct_hamiltonian_NxN(N, Hcell, T, config='Z')
            Ry_Z = np.zeros((2 * N, 2 * N))
            for m in range(N):
                Ry_Z[2 * (N - 1 - m), 2 * m] = 1.0          # Line 1 -> Line 1
                Ry_Z[2 * (N - 1 - m) + 1, 2 * m + 1] = 1.0  # Line 2 -> Line 2
            comm_Z = Ry_Z @ H_Z - H_Z @ Ry_Z
            if np.abs(t1) > 0.1:  # Non-zero hopping
                assert np.max(np.abs(comm_Z)) > 0.1, "Z-configuration should NOT commute with Ry!"

    # 8. Test 2D momentum grid tensor broadcasting for W-configuration
    N_grid = 20
    kx_arr = np.linspace(-k_max, k_max, N_grid)
    ky_arr = np.linspace(-k_max, k_max, N_grid)
    KX, KY = np.meshgrid(kx_arr, ky_arr)
    T1_grid = calc_t1((KX, KY))
    T2_grid = calc_t2((KX, KY))
    Hcell_grid, T_grid = build_cell_and_hopping(T1_grid, T2_grid, eps=epsilon)
    H_tensor_W = construct_hamiltonian_NxN(3, Hcell_grid, T_grid, config='W')
    assert H_tensor_W.shape == (N_grid, N_grid, 7, 7)
    evals_tensor_W = np.linalg.eigvalsh(H_tensor_W)
    assert evals_tensor_W.shape == (N_grid, N_grid, 7)

    print("  [PASS] W-configuration reflection commutator [H_W, J] = 0 (< 1e-14) across all N and k.")
    print("  [PASS] Definite parity eigenvalues lambda = +/- 1.0 verified for all eigenstates.")
    print("  [PASS] Line 1 reflection symmetry p_even[m] = p_even[N-m] (< 1e-14) verified.")
    print("  [PASS] Line 2 reflection symmetry p_odd[m] = p_odd[N-1-m] (< 1e-14) verified.")
    print("  [PASS] Contrast with Z-configuration verified: Z explicitly lacks horizontal reflection symmetry.")
    print("  [PASS] Tensor broadcasting (N_grid, N_grid, 2N+1, 2N+1) verified.")


if __name__ == "__main__":
    print("=" * 65)
    print("MACHINE-PRECISION PARITY SUITE: construct_hamiltonian_NxN(N=2) vs 2xW")
    print("=" * 65)
    test_n2_high_symmetry_points()
    test_n2_full_momentum_grid()
    test_n2_random_k_points()
    test_transverse_profile_normalization()
    test_w_configuration_reflection_symmetry()
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED: Eigenvalues & Transverse Profiles match to machine precision.")
    print("=" * 65)
