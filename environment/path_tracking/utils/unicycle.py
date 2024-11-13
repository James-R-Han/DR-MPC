from copy import deepcopy

import numpy as np

class Unicycle:
    def __init__(self, time_step=0.2, x0=np.zeros(3)):
        self.time_step = time_step
        # poses in global frame
        x0 = np.reshape(x0, (3,1))
        self.x0 = x0
        self.x = deepcopy(x0)
        self.history = deepcopy(x0)
        self.arclengths = [0]
        self.actions = []

    def reset(self):
        self.x = deepcopy(self.x0)
        self.history = deepcopy(self.x0)
        self.arclengths = [0]
        self.actions = []

    def step(self, u, time_step=None):
        time_step = self.time_step if time_step is None else time_step
        delta_x, delta_y, delta_theta = Unicycle.step_calc_global(u, self.x[2,0], time_step)
        self.x[0,0] += delta_x
        self.x[1,0] += delta_y
        self.x[2,0] += delta_theta
        # make sure theta is in [-pi, pi]
        self.x[2,0] = np.mod(self.x[2,0] + np.pi, 2*np.pi) - np.pi
        self.history = np.concatenate((self.history, self.x), axis=1)
        curr_arclength = self.arclengths[-1]
        delta_arclength = time_step*u[0]
        self.arclengths.append(curr_arclength + delta_arclength)
        self.actions.append(u)


    @staticmethod
    def step_external_vectorized(x, u, time_steps):
        # where dts is a numpy array of time steps to evaluate
        num_evals = len(time_steps)
        result = np.zeros((num_evals, 3))

        v, w = u[0], u[1]
        
        if np.abs(w) < 1e-3:
            local_delta_x = v * time_steps
            local_delta_y = np.zeros(num_evals)
            local_delta_theta = np.zeros(num_evals)
        else:
            local_delta_x = (v / w) * np.sin(w * time_steps)
            local_delta_y = (v / w) * (1 - np.cos(w * time_steps))
            local_delta_theta = w * time_steps

        cos_th = np.cos(x[2])
        sin_th = np.sin(x[2])
        rot_G_to_L = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        global_delta_xy = np.matmul(rot_G_to_L, np.array([local_delta_x, local_delta_y]))
        result[:,0] = x[0] + global_delta_xy[0]
        result[:,1] = x[1] + global_delta_xy[1]
        result[:,2] = x[2] + local_delta_theta

        # make sure theta is in [-pi, pi]
        result[:,2] = np.mod(result[:,2] + np.pi, 2*np.pi) - np.pi

        return result
        
    @staticmethod
    def step_local_vectorized(u, time_steps, wrap=True):
        # x,y,th assumed to be 0
        num_evals = len(time_steps)
        result = np.zeros((num_evals, 3))

        v, w = u[0], u[1]
        
        if np.abs(w) < 1e-3:
            result[:,0] = v * time_steps
        else:
            result[:,0] = (v / w) * np.sin(w * time_steps)
            result[:,1] = (v / w) * (1 - np.cos(w * time_steps))
            result[:,2] = w * time_steps
        
            if wrap:
                # make sure theta is in [-pi, pi]
                result[:,2] = np.mod(result[:,2] + np.pi, 2*np.pi) - np.pi

        return result

    @staticmethod
    def step_external(x, u, time_step):
        delta_x, delta_y, delta_theta = Unicycle.step_calc_global(u, x[2], time_step)
        y = x + np.array([delta_x, delta_y, delta_theta])
        # make sure theta is in [-pi, pi]
        y[2] = np.mod(y[2] + np.pi, 2*np.pi) - np.pi
        return y

    @staticmethod
    def step_calc_global(u, th, time_step):
        # relative to [0,0,0]
        delta_x_local, delta_y_local, delta_theta = Unicycle.step_calc_local(u, time_step)
        # convert to global frame
        delta_x, delta_y = Unicycle.local_to_global(delta_x_local, delta_y_local, th)

        return delta_x, delta_y, delta_theta

    @staticmethod
    def local_to_global(delta_x_local, delta_y_local, theta):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        rot_G_to_L = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        delta_x, delta_y = np.matmul(rot_G_to_L, np.array([delta_x_local, delta_y_local]))

        return delta_x, delta_y

    @staticmethod
    def step_calc_local(u, time_step):
        v, w = u[0], u[1]
        if w == 0:
            delta_x = v * time_step
            delta_y = 0
            delta_theta = 0
        else:
            delta_x = (v / w) * np.sin(w * time_step)
            delta_y = (v / w) * (1 - np.cos(w * time_step))
            delta_theta = w * time_step

        return delta_x, delta_y, delta_theta