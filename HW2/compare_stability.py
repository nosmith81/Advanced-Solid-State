"""
Numerical Stability and Method Comparison for HW2:
1. User Formulation: M = eps_bar + t_k @ inv(eta_k), solved with np.linalg.eigvals(M)
2. AI Agent Suggestion: H = eps_bar @ eta_k + t_k, S = eta_k, solved with generalized eigh (Cholesky)
3. True Generalized Eigensolver: A c = lambda B c without assuming Hermiticity
4. Symmetrized Hermitian Formulation: H_sym = 0.5*(eps_bar @ eta_k + eta_k @ eps_bar) + t_k, S = eta_k
"""

import numpy as np
import matplotlib.pyplot as plt
from band_structure import calc_t_matrix, calc_eta_matrix

def solve_user_method(eps_bar, t_k, eta_k):
    """User's current method: invert eta_k and use eigvals."""
    M = eps_bar + t_k @ np.linalg.inv(eta_k)
    evals = np.linalg.eigvals(M)
    return evals

def solve_agent_eigh(eps_bar, t_k, eta_k, lower=True):
    """
    AI Agent's suggestion: H = eps_bar @ eta_k + t_k, S = eta_k.
    Passed to generalized Hermitian eigensolver (scipy.linalg.eigh equivalent).
    Note: H is NOT Hermitian, so eigh will implicitly use lower or upper triangle!
    """
    H = eps_bar @ eta_k + t_k
    S = eta_k
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])
    L_inv = np.linalg.inv(L)
    if lower:
        H_herm = np.tril(H) + np.tril(H, -1).T
    else:
        H_herm = np.triu(H) + np.triu(H, 1).T
    C = L_inv @ H_herm @ L_inv.T
    evals = np.linalg.eigvalsh(C)
    return evals

def solve_general_geneig(eps_bar, t_k, eta_k):
    """
    General generalized eigenvalue problem: A c = lambda B c
    without assuming Hermiticity or positive-definiteness.
    For 2x2, solved exactly via det(A - lambda B) = 0.
    """
    A = eps_bar @ eta_k + t_k
    B = eta_k
    detB = np.linalg.det(B)
    detA = np.linalg.det(A)
    tr_adj = A[0,0]*B[1,1] + A[1,1]*B[0,0] - A[0,1]*B[1,0] - A[1,0]*B[0,1]
    evals = np.roots([detB, -tr_adj, detA])
    return evals

def solve_symmetrized_eigh(eps_bar, t_k, eta_k):
    """
    Physically consistent Hermitian tight-binding formulation:
    H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k.
    This H is manifestly Hermitian.
    """
    H_sym = 0.5 * (eps_bar @ eta_k + eta_k @ eps_bar) + t_k
    S = eta_k
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])
    L_inv = np.linalg.inv(L)
    C = L_inv @ H_sym @ L_inv.T
    evals = np.linalg.eigvalsh(C)
    return evals

def run_path_comparison(a_val, d0=1.0, t0=1.0, num_seg=150):
    k1 = np.linspace([0.0, 0.0], [np.pi, 0.0], num_seg, endpoint=False)
    k2 = np.linspace([np.pi, 0.0], [np.pi, np.pi], num_seg, endpoint=False)
    k3 = np.linspace([np.pi, np.pi], [0.0, 0.0], num_seg + 1, endpoint=True)
    path_ka = np.vstack([k1, k2, k3])
    num_pts = len(path_ka)

    eps_bar = np.array([[0.0, 0.0], [0.0, 0.8 * t0]], dtype=float)
    I2 = np.eye(2, dtype=float)

    R_nn = a_val
    t_nn = calc_t_matrix(R_nn, d0=d0, t0=t0)
    eta_nn = calc_eta_matrix(R_nn, d0=d0)

    R_nnn = np.sqrt(2.0) * a_val
    t_nnn = calc_t_matrix(R_nnn, d0=d0, t0=t0)
    eta_nnn = calc_eta_matrix(R_nnn, d0=d0)

    bands_user_real = np.zeros((num_pts, 2))
    bands_user_imag = np.zeros((num_pts, 2))
    bands_agent_lower = np.zeros((num_pts, 2))
    bands_agent_upper = np.zeros((num_pts, 2))
    bands_geneig_real = np.zeros((num_pts, 2))
    bands_symm = np.zeros((num_pts, 2))
    det_eta = np.zeros(num_pts)
    min_eig_eta = np.zeros(num_pts)

    cholesky_fail_count = 0

    for i in range(num_pts):
        kx, ky = path_ka[i]
        eps_0 = np.cos(kx) + np.cos(ky)
        eps_1 = 2.0 * np.cos(kx) * np.cos(ky)

        t_k = 2.0 * t_nn * eps_0 + 2.0 * t_nnn * eps_1
        eta_k = I2 + 2.0 * eta_nn * eps_0 + 2.0 * eta_nnn * eps_1

        det_eta[i] = np.linalg.det(eta_k)
        eig_eta = np.linalg.eigvalsh(eta_k)
        min_eig_eta[i] = eig_eta[0]

        # 1. User
        ev_u = solve_user_method(eps_bar, t_k, eta_k)
        bands_user_real[i] = np.sort(np.real(ev_u))
        bands_user_imag[i] = np.sort(np.abs(np.imag(ev_u)))

        # 2. Agent lower & upper
        ev_al = solve_agent_eigh(eps_bar, t_k, eta_k, lower=True)
        ev_au = solve_agent_eigh(eps_bar, t_k, eta_k, lower=False)
        if np.isnan(ev_al[0]):
            cholesky_fail_count += 1
        bands_agent_lower[i] = np.sort(ev_al)
        bands_agent_upper[i] = np.sort(ev_au)

        # 3. Exact Gen Eig
        ev_g = solve_general_geneig(eps_bar, t_k, eta_k)
        bands_geneig_real[i] = np.sort(np.real(ev_g))

        # 4. Symmetrized H
        ev_s = solve_symmetrized_eigh(eps_bar, t_k, eta_k)
        bands_symm[i] = np.sort(ev_s)

    path_x = np.linspace(0.0, 3.0, num_pts)
    return {
        'path_x': path_x,
        'bands_user_real': bands_user_real,
        'bands_user_imag': bands_user_imag,
        'bands_agent_lower': bands_agent_lower,
        'bands_agent_upper': bands_agent_upper,
        'bands_geneig_real': bands_geneig_real,
        'bands_symm': bands_symm,
        'det_eta': det_eta,
        'min_eig_eta': min_eig_eta,
        'cholesky_fail_count': cholesky_fail_count,
        'total_pts': num_pts
    }

if __name__ == '__main__':
    print("Running stability and method comparison benchmark...")
    for a in [5.0, 7.0, 9.0]:
        res = run_path_comparison(a)
        print(f"\n================ Lattice constant a = {a} d0 ================")
        print(f"Overlap matrix min eigenvalue: {np.min(res['min_eig_eta']):.4e}")
        print(f"Overlap matrix min determinant: {np.min(res['det_eta']):.4e}")
        print(f"Cholesky failure count (scipy eigh crashes): {res['cholesky_fail_count']}/{res['total_pts']}")
        print(f"Max imaginary part in user's M: {np.max(res['bands_user_imag']):.4e}")
        
        diff_u_gen = np.max(np.abs(res['bands_user_real'] - res['bands_geneig_real']))
        print(f"Max difference |User M - Exact GenEig|: {diff_u_gen:.4e}")
        
        if res['cholesky_fail_count'] == 0:
            diff_agent_l = np.max(np.abs(res['bands_user_real'] - res['bands_agent_lower']))
            diff_agent_u = np.max(np.abs(res['bands_user_real'] - res['bands_agent_upper']))
            diff_l_u = np.max(np.abs(res['bands_agent_lower'] - res['bands_agent_upper']))
            print(f"Agent eigh discrepancy (lower vs upper triangle): {diff_l_u:.4e}")
            print(f"Max discrepancy |User M - Agent eigh(lower)|: {diff_agent_l:.4e}")
            print(f"Max discrepancy |User M - Agent eigh(upper)|: {diff_agent_u:.4e}")
            diff_symm = np.max(np.abs(res['bands_user_real'] - res['bands_symm']))
            print(f"Max difference |User M - Symmetrized H|: {diff_symm:.4e}")
