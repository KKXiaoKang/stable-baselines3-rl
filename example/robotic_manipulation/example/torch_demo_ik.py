import numpy as np

a_dict = np.array([1.0, 1.0, 1.0])
b_dict = np.array([3.0, 3.0, 3.0])

mse_a_eef = np.mean((a_dict - b_dict) ** 2)
print("a_dict - b_dict: ", a_dict - b_dict)
print("mse_a_eef: ", mse_a_eef)