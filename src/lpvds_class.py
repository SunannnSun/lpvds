import numpy as np

""" uncomment the imports below if using DAMM; otherwise import your own methods """
from .damm.damm_class import damm_class
from .ds_opt.dsopt_class import dsopt_class




class lpvds_class():
    def __init__(self, x, x_dot, x_att) -> None:
        self.x      = x
        self.x_dot  = x_dot
        self.x_att  = x_att
        self.dim    = x.shape[1]

        # simulation parameters
        self.tol = 10E-3
        self.max_iter = 10000


    def _cluster(self):
        param ={
            "mu_0":           np.zeros((self.dim, )), 
            "sigma_0":        5 * np.eye(self.dim),
            "nu_0":           self.dim,
            "kappa_0":        1,
            "sigma_dir_0":    1,
            "min_thold":      10
        }

        self.damm  = damm_class(self.x, self.x_dot, param)
        self.gamma = self.damm.begin()

        self.assignment_arr = np.argmax(self.gamma, axis=0)
        self.K     = self.gamma.shape[0]


    def _optimize(self):

        self.ds_opt = dsopt_class(self.x, self.x_dot, self.x_att, self.gamma)
        self.A = self.ds_opt.begin()


    def begin(self):
        self._cluster()
        self._optimize()


    def _step(self, x, dt):
        x_dot     = np.zeros((x.shape[1], 1))

        gamma = self.damm.logProb(x) 
        for k in range(self.K):
            x_dot  += gamma[k, 0] * self.A[k] @ (x - self.x_att).T
        x_next = x + x_dot.T * dt

        return x_next


    def sim(self, x_init, dt):
        x_test = [x_init]

        i = 0
        while np.linalg.norm(x_test[-1]-self.x_att) >= self.tol:
            if i > self.max_iter:
                print("Exceed max iteration")
                break

            x_next = self._step(x_test[-1], dt)
            x_test.append(x_next)        

            i += 1

        return np.vstack(x_test)

