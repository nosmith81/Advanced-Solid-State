import numpy as np
from scipy.optimize import minimize_scalar
from solve_hw1 import objective, generate_hkl, apply_selection_rule, compute_d_obs, load_data, compute_theta

kw = 8.8581
alpha_A, _ = load_data("data.csv")
theta_A = compute_theta(alpha_A)
d_obs = compute_d_obs(theta_A, kw)

hkl_list_all = generate_hkl(6, 6, 6)
hkl_list_sc = apply_selection_rule(hkl_list_all, "all")

eval_points = []
def wrapped_objective(a):
    val = objective(np.array([a]), d_obs, hkl_list_sc, "cubic")
    eval_points.append((a, val))
    return val

res = minimize_scalar(wrapped_objective, bounds=(1.0, 10.0), method='bounded')
print(f"Final result: {res.x}")
print("Evaluation sequence:")
for a, val in eval_points:
    print(f"a = {a:.4f}, val = {val:.2e}")

