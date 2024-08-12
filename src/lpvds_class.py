import os, sys, json
import numpy as np

""" uncomment the imports below if using DAMM; otherwise import your own methods """
from .damm.damm_class import damm_class
from .dsopt.dsopt_class import dsopt_class



def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)




class lpvds_class():
    def __init__(self, x, x_dot, x_att) -> None:
        self.x      = x
        self.x_dot  = x_dot
        self.x_att  = x_att
        self.dim    = 2*x.shape[1]  # either 4 or 6

        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 10000

        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_pos.json')


    def _cluster(self):
        self.param ={
            "mu_0":           np.zeros((self.dim, )), 
            "sigma_0":        0.1 * np.eye(self.dim),
            "nu_0":           self.dim,
            "kappa_0":        0.1,
            "sigma_dir_0":    0.1,
            "min_thold":      10
        }
        
        self.damm  = damm_class(self.x, self.x_dot, self.param)
        self.gamma = self.damm.begin()

        self.assignment_arr = np.argmax(self.gamma, axis=0)
        self.K     = self.gamma.shape[0]


    def _optimize(self):
        self.ds_opt = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        self.A = self.ds_opt.begin()


    def begin(self):
        self._cluster()
        self._optimize()
        # self._logOut()


    def elasticUpdate(self, new_traj, new_gmm_struct):
        x_new, x_dot_new, assignment_arr_new, gamma_new = self.damm.elasticUpdate(new_traj, new_gmm_struct)

        self.ds_opt = dsopt_class(x_new, x_dot_new, self.x_att, gamma_new, assignment_arr_new)
        self.A = self.ds_opt.begin()


    def _step(self, x, dt):
        x_dot     = np.zeros((x.shape[1], 1))

        gamma = self.damm.logProb(x) 
        for k in range(self.K):
            x_dot  += gamma[k, 0] * self.A[k] @ (x - self.x_att).T
        x_next = x + x_dot.T * dt

        return x_next, gamma, x_dot


    def sim(self, x_init, dt):
        x_test = [x_init]
        gamma_test = []
        v_test = []

        i = 0
        while np.linalg.norm(x_test[-1]-self.x_att) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break

            x_next, gamma, v = self._step(x_test[-1], dt)
            x_test.append(x_next)        
            gamma_test.append(gamma[:, 0])
            v_test.append(v)

            i += 1

        return np.vstack(x_test)




    def _logOut(self, *args): 
            Prior = self.damm.Prior
            Mu    = self.damm.Mu
            Sigma = self.damm.Sigma

            json_output = {
                "name": "DAMM-LPVDS",

                "K": self.K,
                "M": Mu.shape[1],
                "Prior": Prior,
                "Mu": Mu.ravel().tolist(),
                "Sigma": Sigma.ravel().tolist(),

                'A': self.A.ravel().tolist(),
                'attractor': self.x_att.ravel().tolist(),
                'att_all': self.x_att.ravel().tolist(),
                "gripper_open": 0
            }

            if len(args) == 0:
                write_json(json_output, self.output_path)
            else:
                write_json(json_output, os.path.join(args[0], '0.json'))




    def begin_next(self, x_new, x_dot_new, x_att_new):
        
        # shift new data
        shift = self.x_att - x_att_new
        x_new_shift = x_new + np.tile(shift, (x_new.shape[0], 1))
        
        # combine batches
        self.x = np.vstack((self.x, x_new_shift))
        self.x_dot = np.vstack((self.x_dot, x_dot_new))

        # construct assignment arr
        comb_assignment_arr = np.concatenate((self.assignment_arr, -1 * np.ones((x_new.shape[0]), dtype=np.int32)))
        
        # run damm
        self.damm  = damm_class(self.x, self.x_dot, self.param)
        self.gamma = self.damm.begin(comb_assignment_arr)
        self.assignment_arr = np.argmax(self.gamma, axis=0)
        self.K     = self.gamma.shape[0]

        # re-learn A
        self._optimize()
        self._logOut()

        # store
        self.x_new_shift = x_new_shift


    def evaluate(self):
        x = self.x
        x_dot = self.x_dot

        x_dot_pred     = np.zeros((x.shape)).T
        gamma = self.damm.logProb(x)
        for k in range(self.K):
            x_dot_pred  += gamma[k, :].reshape(1, -1) * (self.A[k] @ (x - self.x_att).T)

        MSE = np.sum(np.linalg.norm(x_dot_pred-x_dot.T, axis=0))/x.shape[0]
        
        self.x_dot_pred = x_dot_pred
        
        return MSE
    

    def predict(self, x):
        """x has to be [M, N], M is number of points, N is dimension"""
        x_dot_pred     = np.zeros((x.shape)).T

        gamma = self.damm.logProb(x)

        for k in range(self.K):
            x_dot_pred  += gamma[k, :].reshape(1, -1) * (self.A[k] @ (x - self.x_att).T)

        return x_dot_pred