import os
import numpy as np
import matplotlib.pyplot as plt

# Import Packages
from src.util import load_tools, plot_tools
# from src.damm_lpvds import damm_lpvds as damm_lpvds_class

from src.damm.damm_class import damm_class
from src.ds_opt.dsopt_class import dsopt_class


# choose input option
input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. DAMM demo data
Enter the corresponding option number: '''


input_opt = input(input_message)
input_data = load_tools.load_data(int(input_opt))

Data, Data_sh, att, x0_all, dt, _, traj_length = load_tools.processDataStructure(input_data)

# plot_tools.plot_reference_trajectories_DS(Data, att, 100, 20)


dim = 4

param ={
    "mu_0":           np.zeros((dim, )), 
    "sigma_0":        5 * np.eye(dim),
    "nu_0":           dim,
    "kappa_0":        1,
    "sigma_dir_0":    1,
    "min_thold":  10
}

x = Data[:2, :].T
x_dot = Data[2:, :].T
damm = damm_class(x, x_dot, param)      
gamma = damm.begin()
K = damm.K


ds_opt = dsopt_class(x, x_dot, x[-1, :], gamma)
ds_opt.begin()


# damm_lpvds = damm_lpvds_class(Data, Data_sh, att, x0_all, dt, traj_length)



# plt.show()