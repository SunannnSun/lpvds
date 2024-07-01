import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class




i = 0
while True:
    message = 'Press enter to continue: '
    if input(message) == str(0):
        break


    x, x_dot, x_att, x_init = load_tools.load_data('increm')

    if i==0:
        lpvds = lpvds_class(x, x_dot, x_att)
        lpvds.begin()
    else:
        lpvds.begin_next(x, x_dot, x_att)


    # evaluate results
    x_test_list = []
    for x_0 in x_init:
        x_test_list.append(lpvds.sim(x_0, dt=0.01))


    if i == 0:
        plot_tools.plot_ds(x, x_test_list)
    else:
        x_test_list += x_test_prev
        plot_tools.plot_incremental_ds(x, x_prev, x_att, x_test_list)


    # plot results
    plot_tools.plot_gmm(lpvds.x, lpvds.assignment_arr)
    plt.show()

    i+=1
    x_prev = x
    x_test_prev = x_test_list