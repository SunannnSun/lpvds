import numpy as np
from src.util import load_tools


""" uncomment the imports below if using DAMM; otherwise import your own methods """
from src.damm.damm_class import damm_class
from src.ds_opt.dsopt_class import dsopt_class


# choose input option
input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. DAMM demo data
Enter the corresponding option number: '''
input_opt  = input(input_message)

x, x_dot, x_att, x_init = load_tools.load_data(int(input_opt))


dim = 3

param ={
    "mu_0":           np.zeros((dim, )), 
    "sigma_0":        5 * np.eye(dim),
    "nu_0":           dim,
    "kappa_0":        1,
    "sigma_dir_0":    1,
    "min_thold":      10
}


damm  = damm_class(x, x_dot, param)
gamma = damm.begin()


ds_opt = dsopt_class(x, x_dot, x_att, gamma)
ds_opt.begin()
