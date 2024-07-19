import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class


input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''
input_opt  = input(input_message)
# input_opt = 4

x, x_dot, x_att, x_init = load_tools.load_data(int(input_opt))


# run lpvds
lpvds = lpvds_class(x, x_dot, x_att)
lpvds.begin()


# evaluate results
x_test_list = []
for x_0 in x_init:
    x_test_list.append(lpvds.sim(x_0, dt=0.01))


# plot results
plot_tools.plot_gmm(x, lpvds.assignment_arr, lpvds.damm)
if x.shape[1] == 2:
    plot_tools.plot_ds_2d(x, x_test_list, lpvds)
else:
    plot_tools.plot_ds_3d(x, x_test_list)
plt.show()