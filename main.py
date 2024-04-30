import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class


# choose input option
input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. DAMM demo data
Enter the corresponding option number: '''
input_opt  = input(input_message)

x, x_dot, x_att, x_init = load_tools.load_data(int(input_opt))


# run lpvds
lpvds = lpvds_class(x, x_dot, x_att)
lpvds.begin()
x_test = lpvds.sim(x_init[0], dt=0.01)



# plot results
plot_tools.plot_gmm(x, lpvds.assignment_arr)
plot_tools.plot_ds(x, x_test)
plt.show()