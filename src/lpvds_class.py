import os, sys, json, time
import numpy as np

""" uncomment the imports below if using DAMM; otherwise import your own methods """
from .damm.damm_class import damm_class
from .dsopt.dsopt_class import dsopt_class



def _write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)




class lpvds_class():
    def __init__(self, x, x_dot, x_att) -> None:
        self.x      = x
        self.x_dot  = x_dot
        self.x_att  = x_att
        self.x_0    = x[0, :]
        self.dim    = 2*x.shape[1]  # either 4 or 6

        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 10000

        # define output path
        file_path           = os.path.dirname(os.path.realpath(__file__))  
        self.output_path    = os.path.join(os.path.dirname(file_path), 'output_pos.json')

        self.param ={
            "mu_0":           np.zeros((self.dim, )), 
            "sigma_0":        0.01 * np.eye(self.dim),
            "nu_0":           self.dim,
            "kappa_0":        0.01,
            "sigma_dir_0":    0.01,
            "min_thold":      0
        }
        
        self.damm  = damm_class(self.x, self.x_dot, self.param)

    def _cluster(self):
        self.gamma = self.damm.begin()

        # self.assignment_arr = np.argmax(self.gamma, axis=0) # this would result in some component being empty
        # self.K     = self.gamma.shape[0] 

        self.assignment_arr = self.damm.assignment_arr
        self.K = int(self.damm.K)

    def _optimize(self):
        self.ds_opt = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma, self.assignment_arr)
        self.A = self.ds_opt.begin()


    def begin(self):
        self._cluster()
        self._optimize()
        # self._logOut()


    def elasticUpdate(self, new_traj, new_gmm_struct, att_new):
        x_new, x_dot_new, assignment_arr_new, gamma_new = self.damm.elasticUpdate(new_traj, new_gmm_struct)
        self.x_att = att_new
        self.x_0 = x_new[0, :]
        self.K     = gamma_new.shape[0]
        self.ds_opt = dsopt_class(x_new, x_dot_new, att_new, gamma_new, assignment_arr_new)
        self.A = self.ds_opt.begin()

        # self._logOut()


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




    def _logOut(self, write_json=True, *args): 
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
                # 'x_0': self.x[0, :].ravel().tolist(),
                'x_0': self.x_0.ravel().tolist(),

                "gripper_open": 0
            }
            if write_json:
                if len(args) == 0:
                    _write_json(json_output, self.output_path)
                else:
                    _write_json(json_output, os.path.join(args[0], '0.json'))

            return json_output



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