import numpy as np
from solve_hw1 import objective, generate_hkl, apply_selection_rule, compute_d_obs, load_data, compute_theta
import matplotlib.pyplot as plt

kw = 8.8581
alpha_A, _ = load_data("data.csv")
theta_A = compute_theta(alpha_A)
d_obs = compute_d_obs(theta_A, kw)

hkl_list_all = generate_hkl(6, 6, 6)
hkl_list_sc = apply_selection_rule(hkl_list_all, "all")
hkl_list_bcc = apply_selection_rule(hkl_list_all, "bcc")

a_vals = np.linspace(1.0, 10.0, 1000)
obj_sc = [objective(np.array([a]), d_obs, hkl_list_sc, "cubic") for a in a_vals]
obj_bcc = [objective(np.array([a]), d_obs, hkl_list_bcc, "cubic") for a in a_vals]

print(f"Min obj SC at a={a_vals[np.argmin(obj_sc)]}, val={np.min(obj_sc)}")
print(f"Min obj BCC at a={a_vals[np.argmin(obj_bcc)]}, val={np.min(obj_bcc)}")

