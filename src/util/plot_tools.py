import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import random

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.size": 30
})




def plot_reference(Data, att):
    fig = plt.figure(figsize=(8, 6))
    M = len(Data) / 2  # store 1 Dim of Data
    if M == 2:
        ax = fig.add_subplot(111)
        ax.set_xlabel(r'$\xi_1$')
        ax.set_ylabel(r'$\xi_2$')
        ax.set_title('Reference Trajectory')

        # Plot the position trajectories
        plt.plot(Data[0], Data[1], 'ro', markersize=1)
        # plot attractor
        # plt.scatter(att[0], att[1], s=100, c='blue', alpha=0.5)
        plt.scatter(att[0], att[1], marker=(8, 2, 0), s=100, c='k')

        # Plot Velocities of Reference Trajectories
        vel_points = Data[:, ::vel_sample]
        U = np.zeros(len(vel_points[0]))
        V = np.zeros(len(vel_points[0]))  # （385,)
        for i in np.arange(0, len(vel_points[0])):
            dir_ = vel_points[2:, i] / np.linalg.norm(vel_points[2:, i])
            U[i] = dir_[0]
            V[i] = dir_[1]
        q = ax.quiver(vel_points[0], vel_points[1], U, V, width=0.005, scale=vel_size)
    else:
        ax = fig.add_subplot(projection='3d')
        ax.plot(Data[0], Data[1], Data[2], 'ro', markersize=1.5)
        ax.scatter(att[0], att[1], att[2], s=200, c='blue', alpha=0.5)
        ax.axis('auto')
        ax.set_title('Reference Trajectory')
        ax.set_xlabel(r'$\xi_1(m)$')
        ax.set_ylabel(r'$\xi_2(m)$')
        ax.set_zlabel(r'$\xi_3(m)$')
        vel_points = Data[:, ::vel_sample]
        U = np.zeros(len(vel_points[0]))
        V = np.zeros(len(vel_points[0]))
        W = np.zeros(len(vel_points[0]))
        for i in np.arange(0, len(vel_points[0])):
            dir_ = vel_points[3:, i] / np.linalg.norm(vel_points[3:, i])
            U[i] = dir_[0]
            V[i] = dir_[1]
            W[i] = dir_[2]
        q = ax.quiver(vel_points[0], vel_points[1], vel_points[2], U, V, W, length=0.04, normalize=True,colors='k')


    plt.show()




def plot_gmm(x_train, label):
    N = x_train.shape[1]

    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime'] + [
    "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(200)]
    color_mapping = np.take(colors, label)


    fig = plt.figure(figsize=(12, 10))
    if N == 2:
        ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color=color_mapping[:], alpha=0.4, label="Demonstration")

    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color=color_mapping[:], s=3, alpha=0.4, label="Demonstration")
        ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
        ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
        ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='z', which='major', pad=15)




def plot_ds(x_train, x_test_list):
    N = x_train.shape[1]

    fig = plt.figure(figsize=(12, 10))
    if N == 2:
        ax = fig.add_subplot()
        ax.scatter(x_train[:, 0], x_train[:, 1], color='k', s=1, alpha=0.4, label="Demonstration")
        ax.plot(x_test[:, 0], x_test[:, 1], color= 'b')
    elif N == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], 'o', color='k', s=3, alpha=0.4, label="Demonstration")

        for idx, x_test in enumerate(x_test_list):
            ax.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], color= 'b')
        ax.set_xlabel(r'$\xi_1$', fontsize=38, labelpad=20)
        ax.set_ylabel(r'$\xi_2$', fontsize=38, labelpad=20)
        ax.set_zlabel(r'$\xi_3$', fontsize=38, labelpad=20)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(axis='z', which='major', pad=15)
