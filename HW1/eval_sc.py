import numpy as np
from solve_hw1 import load_data, compute_theta, compute_sin2theta, objective, generate_hkl, apply_selection_rule

kw = 8.8581
alpha_A, _ = load_data("data.csv")
theta_obs = compute_theta(alpha_A)
sin2theta_obs = compute_sin2theta(theta_obs)

hkl_list_all = generate_hkl(6, 6, 6)
hkl_list = apply_selection_rule(hkl_list_all, "all")

a = 2.026561 
R = objective(np.array([a]), sin2theta_obs, hkl_list, "cubic", kw)
print(f"SC with a={a} -> R = {R}")
