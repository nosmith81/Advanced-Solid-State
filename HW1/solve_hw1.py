import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import os

############ Data Loading

def load_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    return df['A'].values, df['B'].values

def compute_theta(alpha: np.ndarray) -> np.ndarray:
    return alpha / 2.0

def compute_d_obs(theta: np.ndarray, kw: float) -> np.ndarray:
    return np.pi / (kw * np.sin(theta))

####### Miller Index Generation

def generate_hkl(h_max: int, k_max: int, l_max: int) -> list[tuple[int, int, int]]:
    hkl_list = []
    for h in range(h_max + 1):
        for k in range(k_max + 1):
            for l in range(l_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl_list.append((h, k, l))
    return hkl_list

def apply_selection_rule(hkl_list: list[tuple[int, int, int]], rule: str) -> list[tuple[int, int, int]]:
    filtered_list = []
    for h, k, l in hkl_list:
        if rule == "all":
            filtered_list.append((h, k, l))
        elif rule == "bcc":
            if (h + k + l) % 2 == 0:
                filtered_list.append((h, k, l))
        elif rule == "fcc":
            is_odd = [i % 2 != 0 for i in (h, k, l)]
            if all(is_odd) or not any(is_odd):
                filtered_list.append((h, k, l))
        elif rule == "hcp":
            # HCP selection rule: absent if (h + 2k) is a multiple of 3 AND l is odd
            if not ((h + 2 * k) % 3 == 0 and l % 2 != 0):
                filtered_list.append((h, k, l))
        else:
            raise ValueError(f"Unknown rule: {rule}")
    return filtered_list

############## Predicted d-spacing Calculators

def d_spacing_cubic(h: int, k: int, l: int, a: float) -> float:
    return a / np.sqrt(h**2 + k**2 + l**2)

def d_spacing_tetragonal(h: int, k: int, l: int, a: float, c: float) -> float:
    return 1.0 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)

def d_spacing_hexagonal(h: int, k: int, l: int, a: float, c: float) -> float:
    return 1.0 / np.sqrt((4.0 / 3.0) * (h**2 + h*k + k**2) / a**2 + l**2 / c**2)

############## Objective Function and Minimization

def compute_all_predicted_d(
    params: np.ndarray,
    hkl_list: list[tuple[int, int, int]],
    lattice_type: str
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    
    d_list = []
    for h, k, l in hkl_list:
        if lattice_type == "cubic":
            a = params[0]
            val = d_spacing_cubic(h, k, l, a)
        elif lattice_type == "tetragonal":
            a, c = params[0], params[1]
            val = d_spacing_tetragonal(h, k, l, a, c)
        elif lattice_type == "hexagonal":
            a, c = params[0], params[1]
            val = d_spacing_hexagonal(h, k, l, a, c)
        else:
            raise ValueError(f"Unknown lattice_type: {lattice_type}")
        d_list.append((val, (h, k, l)))
    
    # Sort and remove duplicates based on d value (within some tolerance)
    # Sort in descending order (largest d first = smallest theta)
    d_list.sort(key=lambda x: x[0], reverse=True)
    
    unique_d = []
    unique_hkl = []
    
    for val, hkl in d_list:
        if not unique_d or abs(val - unique_d[-1]) > 1e-10:
            unique_d.append(val)
            unique_hkl.append(hkl)
            
    return np.array(unique_d), unique_hkl

def match_peaks(
    d_obs: np.ndarray,
    d_pred: np.ndarray,
    hkl_sorted: list[tuple[int, int, int]]
) -> list[dict]:
    
    matches = []
    for i, d_o in enumerate(d_obs):
        diffs = np.abs(d_pred - d_o)
        best_idx = np.argmin(diffs)
        
        matches.append({
            "peak_index": i,
            "hkl": hkl_sorted[best_idx],
            "d_obs": d_o,
            "d_pred": d_pred[best_idx],
            "residual": diffs[best_idx]
        })
        
    return matches

def objective(
    params: np.ndarray,
    d_obs: np.ndarray,
    hkl_list: list[tuple[int, int, int]],
    lattice_type: str
) -> float:
    
    d_pred, hkl_sorted = compute_all_predicted_d(params, hkl_list, lattice_type)
    
    residual_sq_sum = 0.0
    
    # 1. Forward match: for each observed peak, find closest predicted
    for d_o in d_obs:
        diffs = np.abs(d_pred - d_o)
        residual_sq_sum += np.min(diffs)**2
        
    # 2. Backward match: for each predicted peak in range, find closest observed
    # This penalizes predicting peaks that don't exist in the data
    d_min, d_max = np.min(d_obs), np.max(d_obs)
    for d_p in d_pred:
        # Include a tiny tolerance to account for floating point matching at the boundaries
        if (d_min - 1e-5) <= d_p <= (d_max + 1e-5):
            diffs = np.abs(d_obs - d_p)
            residual_sq_sum += np.min(diffs)**2
            
    return residual_sq_sum

def fit_lattice(
    d_obs: np.ndarray,
    lattice_type: str,
    selection_rule: str,
    h_max: int = 6,
    k_max: int = 6,
    l_max: int = 6,
    a_bounds: tuple[float, float] = (1.0, 10.0),
    c_bounds: tuple[float, float] = (1.0, 10.0),
    n_grid: int = 10
) -> dict:
    
    hkl_list_all = generate_hkl(h_max, k_max, l_max)
    hkl_list = apply_selection_rule(hkl_list_all, selection_rule)
    
    if lattice_type == "cubic":
        best_params = None
        best_residual = float('inf')
        
        # Use a finer grid for 1D to ensure we don't miss the global minimum
        a_vals = np.linspace(a_bounds[0], a_bounds[1], n_grid * 5)
        
        for a0 in a_vals:
            res = minimize(
                objective,
                x0=np.array([a0]),
                args=(d_obs, hkl_list, lattice_type),
                bounds=[a_bounds],
                method='L-BFGS-B'
            )
            if res.fun < best_residual:
                best_residual = res.fun
                best_params = res.x
    else:
        best_params = None
        best_residual = float('inf')
        
        a_vals = np.linspace(a_bounds[0], a_bounds[1], n_grid)
        c_vals = np.linspace(c_bounds[0], c_bounds[1], n_grid)
        
        for a0 in a_vals:
            for c0 in c_vals:
                res = minimize(
                    objective,
                    x0=np.array([a0, c0]),
                    args=(d_obs, hkl_list, lattice_type),
                    bounds=[a_bounds, c_bounds],
                    method='L-BFGS-B'
                )
                if res.fun < best_residual:
                    best_residual = res.fun
                    best_params = res.x
                    
    label_map = {
        ("cubic", "all"): "SC",
        ("cubic", "bcc"): "BCC",
        ("cubic", "fcc"): "FCC",
        ("tetragonal", "all"): "P-Tet",
        ("tetragonal", "bcc"): "BC-Tet",
        ("hexagonal", "all"): "Hex",
        ("hexagonal", "hcp"): "HCP"
    }
    label = label_map.get((lattice_type, selection_rule), "Unknown")
    
    return {
        "params": best_params,
        "residual": best_residual,
        "lattice_type": lattice_type,
        "selection_rule": selection_rule,
        "label": label,
        "hkl_list": hkl_list
    }

def fit_all_lattices(
    d_obs: np.ndarray
) -> list[dict]:
    
    lattice_configs = [
        ("cubic", "all"),
        ("cubic", "bcc"),
        ("cubic", "fcc"),
        ("tetragonal", "all"),
        ("tetragonal", "bcc"),
        ("hexagonal", "all"),
        ("hexagonal", "hcp")
    ]
    
    results = []
    for lat_type, sel_rule in lattice_configs:
        res = fit_lattice(d_obs, lat_type, sel_rule)
        results.append(res)
        
    return results

# ########### Plotting

def plot_lattice_fit(
    theta_obs: np.ndarray,
    fit_result: dict,
    kw: float,
    crystal_label: str,
    save_path: str | None = None
) -> plt.Figure:
    
    d_obs = compute_d_obs(theta_obs, kw)
    d_pred, hkl_sorted = compute_all_predicted_d(
        fit_result["params"], fit_result["hkl_list"], fit_result["lattice_type"]
    )
    
    matches = match_peaks(d_obs, d_pred, hkl_sorted)
    
    # We want theta, not d
    theta_pred_matched = np.arcsin(np.clip(np.pi / (kw * np.array([m["d_pred"] for m in matches])), -1, 1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(theta_obs))
    x_labels = [str(m["hkl"]) for m in matches]
    
    ax.plot(x_positions, theta_obs, 'o-', color='blue', label='Observed $\\theta$')
    ax.plot(x_positions, theta_pred_matched, 'x--', color='red', label='Predicted $\\theta$')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    ax.set_xlabel("(hkl)")
    ax.set_ylabel("$\\theta$ (rad)")
    
    res = fit_result["residual"]
    label = fit_result["label"]
    ax.set_title(f"Crystal {crystal_label} — {label} fit (R = {res:.2e})")
    
    ax.legend()
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        
    return fig

def plot_all_fits(
    theta_obs: np.ndarray,
    all_results: list[dict],
    kw: float,
    crystal_label: str,
    save_dir: str
) -> None:
    
    os.makedirs(save_dir, exist_ok=True)
    
    for res in all_results:
        label = res["label"]
        save_path = os.path.join(save_dir, f"crystal_{crystal_label}_{label}.png")
        fig = plot_lattice_fit(theta_obs, res, kw, crystal_label, save_path)
        plt.close(fig)


def main() -> None:
    kw = 8.8581
    
    alpha_A, alpha_B = load_data("data2.csv")
    
    for crystal_label, alpha in [("A", alpha_A), ("B", alpha_B)]:
        print(f"\n{'='*50}")
        print(f"Analyzing Crystal {crystal_label}")
        print(f"{'='*50}")
        
        theta_obs = compute_theta(alpha)
        d_obs = compute_d_obs(theta_obs, kw)
        
        all_results = fit_all_lattices(d_obs)
        all_results.sort(key=lambda x: x["residual"])
        
        print(f"\nSummary for Crystal {crystal_label}:")
        print(f"{'Label':<10} | {'Lattice Type':<12} | {'a (A)':<8} | {'c (A)':<8} | {'Residual R'}")
        print("-" * 65)
        for res in all_results:
            a = res["params"][0]
            c = res["params"][1] if len(res["params"]) > 1 else np.nan
            print(f"{res['label']:<10} | {res['lattice_type']:<12} | {a:8.4f} | {c:8.4f} | {res['residual']:.2e}")
            
        best = all_results[0]
        
        a_best = best["params"][0]
        if len(best["params"]) > 1:
            c_best = best["params"][1]
            print(f"\nBest fit: {best['label']} with a = {a_best:.4f}, c = {c_best:.4f}, R = {best['residual']:.2e}")
        else:
            print(f"\nBest fit: {best['label']} with a = {a_best:.4f}, R = {best['residual']:.2e}")
            
        d_pred, hkl_sorted = compute_all_predicted_d(
            best["params"], best["hkl_list"], best["lattice_type"]
        )
        matches = match_peaks(d_obs, d_pred, hkl_sorted)
        
        print(f"\nPer-peak match for {best['label']}:")
        print(f"{'Peak':<5} | {'(hkl)':<12} | {'theta_obs (rad)':<15} | {'theta_pred (rad)':<16} | {'delta theta (rad)'}")
        print("-" * 75)
        for m in matches:
            t_obs = np.arcsin(np.clip(np.pi / (kw * m["d_obs"]), -1, 1))
            t_pred = np.arcsin(np.clip(np.pi / (kw * m["d_pred"]), -1, 1))
            dt = abs(t_obs - t_pred)
            print(f"{m['peak_index']+1:<5} | {str(m['hkl']):<12} | {t_obs:<15.6f} | {t_pred:<16.6f} | {dt:.2e}")
            
        plot_all_fits(theta_obs, all_results, kw, crystal_label, "plots")

if __name__ == "__main__":
    main()
