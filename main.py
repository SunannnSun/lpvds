import sys, os
import numpy as np
from scipy.io import loadmat

dir_path     = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, 'damm'))
sys.path.append(os.path.join(dir_path, 'ds_opt'))


from damm.main import damm
from ds_opt.utils_ds import load_tools
from ds_opt.main import DsOpt

# Define Path
input_path    = os.path.join(dir_path, 'input.mat')
output_path   = os.path.join(dir_path, 'output.json')


dim = 2
# Load Data
if dim ==2:
    data_ = loadmat(r"{}".format(input_path))
    data_ = np.array(data_["data"])
    N = len(data_[0])
    input_data = data_.reshape((N, 1))
else:
    data_ = loadmat(r"{}".format(input_path))
    data_ = np.array(data_["data"])
    N = len(data_)
    # traj = np.random.choice(np.arange(N), 4, replace=False)
    traj = np.array([6, 8, 3, 5]) - 1
    input_data = data_[traj]

Data, Data_sh, att, x0_all, dt, _, traj_length = load_tools.processDataStructure(input_data)


# Init dict
data_dict = {
    "Data": Data,
    "Data_sh": Data_sh,
    "att": att,
    "x0_all": x0_all,
    "dt": dt,
    "traj_length":traj_length
}


# Run DAMM
DAMM = damm(Data)         
if DAMM.begin() == 0:
    DAMM.result(if_plot=True)


# Run ds_opt and output JSON
ds_opt = DsOpt(data_dict, os.path.join(dir_path, "output.json"))
ds_opt.begin()
ds_opt.evaluate()
ds_opt.make_plot()
