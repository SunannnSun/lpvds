import numpy as np
import matplotlib.pyplot as plt

from src.util import load_tools, plot_tools
from src.lpvds_class import lpvds_class
import matlab.engine



eng = matlab.engine.start_matlab()
eng.cd(r'../Developers/rosbag_to_mat')


i = 0
while True:
    message = 'Press enter to continue: '
    if input(message) == str(0):
        break


    if i==0:
        x, x_dot, x_att, x_init = load_tools.load_data('demo')
        lpvds = lpvds_class(x, x_dot, x_att)
        lpvds.begin()
    else:
        eng.process_rosbags(nargout=0)
        x, x_dot, x_att, x_init = load_tools.load_data('increm')
        lpvds.begin_next(x, x_dot, x_att)


    # evaluate results
    x_test_list = []
    for x_0 in x_init:
        x_test_list.append(lpvds.sim(x_0, dt=0.01))


    if i == 0:
        plot_tools.plot_ds_3d(x, x_test_list)
    else:
        x_test_list += x_test_prev
        plot_tools.plot_incremental_ds(lpvds.x_new_shift, x_prev, lpvds.x_att, x_test_list)


    # plot results
    plot_tools.plot_gmm(lpvds.x, lpvds.assignment_arr, lpvds.damm)
    plt.show()

    i+=1
    x_prev = x
    x_test_prev = x_test_list