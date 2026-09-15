import numpy as np
from solve_hw1 import objective, generate_hkl, apply_selection_rule, compute_d_obs, load_data, compute_theta
import matplotlib.pyplot as plt

kw = 8.8581
alpha_A, _ = load_data("data.csv")
theta_A = compute_theta(alpha_A)
d_obs = compute_d_obs(theta_A, kw)

hkl_list_all = generate_hkl(6, 6, 6)
hkl_list_sc = apply_selection_rule(hkl_list_all, "all")

a_vals = np.linspace(1.0, 10.0, 5000)
obj_sc = [objective(np.array([a]), d_obs, hkl_list_sc, "cubic") for a in a_vals]

for i in range(1, len(a_vals)-1):
    if obj_sc[i] < obj_sc[i-1] and obj_sc[i] < obj_sc[i+1] and obj_sc[i] < 0.01:
        print(f"Local min at a={a_vals[i]:.4f}, val={obj_sc[i]:.2e}")

