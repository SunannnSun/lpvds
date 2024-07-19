import os, sys
import numpy as np
import pyLasaDataset as lasa
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


def load_data(input_opt):
    """
    Return:
    -------
        x:     a [M, N] NumPy array: M observations of N dimension
    
        x_dot: a [M, N] NumPy array: M observations velocities of N dimension

        x_att: a [1, N] NumPy array of attractor

        x_init: an L-length list of [1, N] NumPy array: L number of trajectories, each containing an initial point of N dimension
    """

    if input_opt == 1:
        print("\nYou selected PC-GMM benchmark data.\n")
        pcgmm_list = ["3D_sink", "3D_viapoint_1", "3D-cube-pick", "3D_viapoint_2", "2D_Lshape",  "2D_incremental_1", "2D_multi-behavior", "2D_messy-snake"]

        message = """Available Models: \n"""
        for i in range(len(pcgmm_list)):
            message += "{:2}) {: <18} ".format(i+1, pcgmm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '

        data_opt = int(input(message))
        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt>len(pcgmm_list):
            print("Invalid data option")
            sys.exit()

        data_name  = str(pcgmm_list[data_opt-1]) + ".mat"
        input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "pc-gmm-data", data_name)

        data_ = loadmat(r"{}".format(input_path))
        data_ = np.array(data_["data"])

        N     = int(data_[0, 0].shape[0]/2)
        if N == 2:
            L = data_.shape[1]
            x     = [data_[0, l][:N, :].T  for l in range(L)]
            x_dot = [data_[0, l][N:, :].T  for l in range(L)]
        elif N == 3:
            L = data_.shape[0]
            L_sub = np.random.choice(range(L), 6, replace=False)

            x     = [data_[l, 0][:N, :].T  for l in range(L)]
            x_dot = [data_[l, 0][N:, :].T  for l in range(L)]


    elif input_opt == 2:
        print("\nYou selected LASA benchmark dataset.\n")

        # suppress print message from lasa package
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')
        sys.stdout = original_stdout

        lasa_list = ["Angle", "BendedLine", "CShape", "DoubleBendedLine", "GShape", "heee", "JShape", "JShape_2", "Khamesh", "Leaf_1",
        "Leaf_2", "Line", "LShape", "NShape", "PShape", "RShape", "Saeghe", "Sharpc", "Sine", "Snake",
        "Spoon", "Sshape", "Trapezoid", "Worm", "WShape", "Zshape", "Multi_Models_1", "Multi_Models_2", "Multi_Models_3", "Multi_Models_4"]

        message = """Available Models: \n"""
        for i in range(len(lasa_list)):
            message += "{:2}) {: <18} ".format(i+1, lasa_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))

        if data_opt == 0:
            sys.exit()
        elif data_opt<0 or data_opt > len(lasa_list):
            print("Invalid data option")
            sys.exit()

        data = getattr(lasa.DataSet, lasa_list[data_opt-1])
        demos = data.demos 
        sub_sample = 1
        L = len(demos)

        x     = [demos[l].pos[:, ::sub_sample].T for l in range(L)]
        x_dot = [demos[l].vel[:, ::sub_sample].T for l in range(L)]


    elif input_opt == 3:
        print("\nYou selected Damm demo dataset.\n")

        damm_list = ["bridge", "Nshape", "orientation"]
        
        message = """Available Models: \n"""
        for i in range(len(damm_list)):
            message += "{:2}) {: <18} ".format(i+1, damm_list[i])
            if (i+1) % 6 ==0:
                message += "\n"
        message += '\nEnter the corresponding option number [type 0 to exit]: '
        
        data_opt = int(input(message))
    
        folder_name = str(damm_list[data_opt-1])
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "damm-demo-data", folder_name, "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 4:
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "obstacle", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'demo':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "demo", "all.mat")
        x, x_dot    = _process_bag(input_path)



    elif input_opt == 'increm':
        print("\nYou selected demo.\n")
        input_path  = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..", "..", "dataset", "increm", "all.mat")
        x, x_dot    = _process_bag(input_path)



    return _pre_process(x, x_dot)





def _pre_process(x, x_dot):
    """ 
    Roll out nested lists into a single list of M entries

    Parameters:
    -------
        x:     an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations of N dimension,
    
        x_dot: an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations velocities of N dimension

    Note:
    -----
        M can vary and need not be same between trajectories
    """

    L = len(x)
    x_init = []
    x_shifted = []

    x_att  = [x[l][-1, :]  for l in range(L)]  
    x_att_mean  =  np.mean(np.array(x_att), axis=0, keepdims=True)
    for l in range(L):
        x_init.append(x[l][0].reshape(1, -1))

        x_diff = x_att_mean - x_att[l]
        x_shifted.append(x_diff.reshape(1, -1) + x[l])

    for l in range(L):
        if l == 0:
            x_rollout = x_shifted[l]
            x_dot_rollout = x_dot[l]
        else:
            x_rollout = np.vstack((x_rollout, x_shifted[l]))
            x_dot_rollout = np.vstack((x_dot_rollout, x_dot[l]))

    return  x_rollout, x_dot_rollout, x_att_mean, x_init




def _process_bag(path):
    """ Process .mat files that is converted from .bag files """

    data_ = loadmat(r"{}".format(path))
    data_ = data_['data_ee_pose']
    L = data_.shape[1]

    x     = []
    x_dot = [] 

    sample_step = 5
    vel_thresh  = 1e-3 
    
    for l in range(L):
        data_l = data_[0, l]['pose'][0,0]
        pos_traj  = data_l[:3, ::sample_step]
        quat_traj = data_l[3:7, ::sample_step]
        time_traj = data_l[-1, ::sample_step].reshape(1,-1)

        raw_diff_pos = np.diff(pos_traj)
        vel_mag = np.linalg.norm(raw_diff_pos, axis=0).flatten()
        first_non_zero_index = np.argmax(vel_mag > vel_thresh)
        last_non_zero_index = len(vel_mag) - 1 - np.argmax(vel_mag[::-1] > vel_thresh)

        if first_non_zero_index >= last_non_zero_index:
            raise Exception("Sorry, vel are all zero")

        pos_traj  = pos_traj[:, first_non_zero_index:last_non_zero_index]
        quat_traj = quat_traj[:, first_non_zero_index:last_non_zero_index]
        time_traj = time_traj[:, first_non_zero_index:last_non_zero_index]
        vel_traj = np.diff(pos_traj) / np.diff(time_traj)
        
        x.append(pos_traj[:, 0:-1].T)
        x_dot.append(vel_traj.T)

    return x, x_dot