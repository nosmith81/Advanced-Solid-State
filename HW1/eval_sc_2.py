import numpy as np
from solve_hw1 import load_data, compute_theta, compute_sin2theta, objective, generate_hkl, apply_selection_rule, compute_all_predicted_sin2theta, match_peaks

kw = 8.8581
alpha_A, _ = load_data("data.csv")
theta_obs = compute_theta(alpha_A)
sin2theta_obs = compute_sin2theta(theta_obs)

hkl_list_all = generate_hkl(6, 6, 6)
hkl_list = apply_selection_rule(hkl_list_all, "all")

a = 2.026561
sin2_pred, hkl_sorted = compute_all_predicted_sin2theta(np.array([a]), hkl_list, "cubic", kw)
matches = match_peaks(sin2theta_obs, sin2_pred, hkl_sorted)
for m in matches:
    print(f"Peak {m['peak_index']+1}: {m['hkl']} obs={m['sin2theta_obs']:.6f} pred={m['sin2theta_pred']:.6f} res={m['residual']:.2e}")
