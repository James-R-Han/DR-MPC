import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from environment.path_tracking.utils.unicycle import Unicycle

from pylgmath import Transformation
from pysteam.pysteam.problem import OptimizationProblem, WeightedLeastSquareCostTerm, StaticNoiseModel, L2LossFunc, CauchyLossFunc, L2LossFuncPose
from pysteam.pysteam.evaluable.se3 import SE3StateVar
from pysteam.pysteam.evaluable.vspace import VSpaceStateVar
from pysteam.pysteam.evaluable import Evaluable, Node
from pysteam.pysteam.evaluable.stereo import ComposeLandmarkEvaluator, HomoPointStateVar, HomoPointErrorEvaluator, HomoPointErrorEvaluator2, HomoPointErrorEvaluator3, HomoPointErrorEvaluator4
from pysteam.pysteam.evaluable.se3.se3_evaluators import LogMapEvaluator, SE3ErrorEvaluator, ExpMapEvaluator, InverseEvaluator, ComposeEvaluator, ComposeInverseEvaluator, SE3LateralEvaluator, LogEvaluator, SE3LateralErrorEvaluator
from pysteam.pysteam.evaluable.vspace.vspace_evaluators import NegationEvaluator, ScalarAdditionEvaluator, AdditionEvaluator, ScalarMultEvaluator, MatrixMultEvaluator, VSpaceErrorEvaluator, MatrixMultEvaluatorRHS, MatrixMultEvaluatorTran, ScalarInverseEvaluator, ScalarLogBarrierEvaluator
from pysteam_augmented.lev_marq_gauss_newton_custom_solver import LevMarqGaussNewtonCustomSolver


class MPC:

    def __init__(self, config_master, K, time_step, continuous_task, max_v, max_w):
        self.config_master = config_master
        self.K = K
        self.time_step = time_step
        self.continuous_task = continuous_task
    
        self.max_v_scalar, self.max_w_scalar = max_v, max_w
        # In this specific MPC formulation, we want the mpc target speed to be slightly lower than max robot speed. If robot is off the path and mpc speed = max speed, then it is difficult to recover. Even with this slight sacrifice, DR-MPC outperforms other methods.
        self.travel_dist_in_DT = (self.max_v_scalar * 0.99) * self.time_step 
        self.node_idx_increment = self.travel_dist_in_DT/self.config_master.config_PT.env.node_separation

        self.P_tran = -1 * np.array([[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
        self.P_tran_v = np.array([[1, 0]])
        self.P_tran_w = np.array([[0, 1]])

        self.action_extra_before = np.array([[1, 0 ,0 ,0]])
        self.action_extra_after = np.array([[1, 0 ,0 ,0]]).T

        self.pose_error_scaling_base = 1
        self.theta_error_scaling_base = 1
        self.base_noise_model_pose_arr = np.eye(6)*self.pose_error_scaling_base
        self.noise_model_kin = StaticNoiseModel(0.001 * np.eye(6))
        self.noise_linvel_bound = StaticNoiseModel(np.array([[10]]))
        self.noise_angvel_bound = StaticNoiseModel(np.array([[10]]))

        self.max_v = VSpaceStateVar(np.array([[self.max_v_scalar]]), locked=True)
        self.min_v = VSpaceStateVar(np.array([[0.0]]), locked=True)
        self.max_w = VSpaceStateVar(np.array([[self.max_w_scalar]]), locked=True)
        self.min_w = VSpaceStateVar(np.array([[-self.max_w_scalar]]), locked=True)

        self.iteration0 = True
        self.next_vspace_state_vars = 0
        self.next_se3_state_vars = 0

    def reset(self):
        self.iteration0 = True

    def act(self, state_gen_input):
        path = state_gen_input['path']
        interp_index = state_gen_input['interp_index']
        curr_pose = state_gen_input['pose']

        solved = False
        pose_error_scaling = 1
        while solved is False:
            self.noise_model_pose = StaticNoiseModel(self.base_noise_model_pose_arr*pose_error_scaling)
            vspace_state_vars = []
            if self.iteration0:
                for i in range(self.K):
                    vspace_state_vars.append(VSpaceStateVar(np.array([[self.max_v_scalar/2], [0]])))
                se3_state_vars, ref_poses = self.generate_local_ref_path(path, interp_index, se3_state_var=True)
                self.iteration0 = False
            else:
                vspace_state_vars = deepcopy(self.next_vspace_state_vars)
                se3_state_vars = deepcopy(self.next_se3_state_vars) # NEED to deepcopy because of try except and we append to a list

            # override current first pose with current pose
            curr_T_ba_array = np.array([[np.cos(curr_pose[2]), -np.sin(curr_pose[2]), 0, curr_pose[0]],
                                [np.sin(curr_pose[2]), np.cos(curr_pose[2]), 0, curr_pose[1]],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

            se3_state_vars.insert(0, SE3StateVar(Transformation(T_ba=curr_T_ba_array).inverse()))
            se3_state_vars[0].locked = True
            measurements, ref_poses = self.generate_local_ref_path(path, interp_index, se3_state_var=False)

            # setup optimization problem
            opt_prob = OptimizationProblem()
            for i in range(1, self.K+1, 1):  # note that locked state variables should not be added to the optimization problem, nor should they be locked after having added them to the optimization problem
                opt_prob.add_state_var(se3_state_vars[i])

            for i in range(self.K):
                opt_prob.add_state_var(vspace_state_vars[i])

            loss_func = L2LossFunc()  # Default least squares function, constant for all cost terms
            loss_func_pose = L2LossFuncPose()

            # Generate cost terms
            for i in range(self.K):

                # Pose Error Cost Term using the builtin SE3 error
                pose_error_func = SE3ErrorEvaluator(se3_state_vars[i+1], measurements[i]) # +1 because we inserted initial pose
                # Pose Error cost term using chained evaluators
                T_cost_term = WeightedLeastSquareCostTerm(pose_error_func, self.noise_model_pose, loss_func_pose)
                opt_prob.add_cost_term(T_cost_term)


                lhs = ComposeInverseEvaluator(se3_state_vars[i + 1], se3_state_vars[i])
                rhs = ExpMapEvaluator(ScalarMultEvaluator(MatrixMultEvaluator(vspace_state_vars[i], self.P_tran), self.time_step))
                kin_error_func = LogMapEvaluator(ComposeInverseEvaluator(lhs, rhs))
                kin_cost_term = WeightedLeastSquareCostTerm(kin_error_func, self.noise_model_kin, loss_func)
                opt_prob.add_cost_term(kin_cost_term)

                # log barriers for velocity bounds. Upper and lower for v,w --> 4 cost terms
                v_w = vspace_state_vars[i]
                v = MatrixMultEvaluator(v_w, self.P_tran_v)
                v_upper_diff = AdditionEvaluator(self.max_v, NegationEvaluator(v))
                v_upper_diff_normalized = ScalarMultEvaluator(v_upper_diff, 1/self.max_v_scalar)
                v_upper_error_func = LogEvaluator(v_upper_diff_normalized)
                v_upper_error_func_scaled = ScalarMultEvaluator(v_upper_error_func, 1/500)
                v_upper_cost_term = WeightedLeastSquareCostTerm(v_upper_error_func_scaled, self.noise_linvel_bound, loss_func)
                v_lower_diff = AdditionEvaluator(v, NegationEvaluator(self.min_v))
                v_lower_diff_normalized = ScalarMultEvaluator(v_lower_diff, 1/self.max_v_scalar)
                v_lower_error_func = LogEvaluator(v_lower_diff_normalized)
                v_lower_error_func_scaled = ScalarMultEvaluator(v_lower_error_func, 1/500)
                v_lower_cost_term = WeightedLeastSquareCostTerm(v_lower_error_func_scaled, self.noise_linvel_bound, loss_func)
                opt_prob.add_cost_term(v_upper_cost_term)
                opt_prob.add_cost_term(v_lower_cost_term)

                w = MatrixMultEvaluator(v_w, self.P_tran_w)
                w_upper_diff = AdditionEvaluator(self.max_w, NegationEvaluator(w))
                w_upper_diff_normalized = ScalarMultEvaluator(w_upper_diff, 1/(2*self.max_w_scalar))
                w_upper_error_func = LogEvaluator(w_upper_diff_normalized)
                w_upper_error_func_scaled = ScalarMultEvaluator(w_upper_error_func, 1/500)
                w_upper_cost_term = WeightedLeastSquareCostTerm(w_upper_error_func_scaled, self.noise_angvel_bound, loss_func)
                w_lower_diff = AdditionEvaluator(w, NegationEvaluator(self.min_w))
                w_lower_diff_normalized = ScalarMultEvaluator(w_lower_diff, 1/(2*self.max_w_scalar))
                w_lower_error_func = LogEvaluator(w_lower_diff_normalized)
                w_lower_error_func_scaled = ScalarMultEvaluator(w_lower_error_func, 1/500)
                w_lower_cost_term = WeightedLeastSquareCostTerm(w_lower_error_func_scaled, self.noise_angvel_bound, loss_func)
                opt_prob.add_cost_term(w_upper_cost_term)
                opt_prob.add_cost_term(w_lower_cost_term)

            solver = LevMarqGaussNewtonCustomSolver(opt_prob, verbose=False, max_iterations=1500, relative_cost_change_threshold=1e-4, v_max=self.max_v_scalar, v_min=0, w_max=self.max_w_scalar, w_min=-self.max_w_scalar)
            try:
                solver.optimize()
                applied_vels = np.concatenate([np.squeeze(v.value, 1) for v in vspace_state_vars], axis=0)
                solved = True
    
                if pose_error_scaling > 500:
                    # useful when I was debugging
                    th = path[2, :]
                    # convert th to dx,dy vector
                    dx = np.cos(th)
                    dy = np.sin(th)
                    fig, ax = plt.subplots()
                    plt.quiver(path[0, :], path[1, :], dx, dy, label='poses', color='g')
                    plt.quiver(curr_pose[0], curr_pose[1], np.cos(curr_pose[2]), np.sin(curr_pose[2]), color='r',
                            label='curr point with heading')
                    plt.scatter(curr_pose[0], curr_pose[1], c='r', label='curr point')

                    ref_poses = np.stack(ref_poses, axis=1)
                    th_ref_poses = ref_poses[2, :]
                    # convert th to dx,dy vector
                    dx_ref_poses = np.cos(th_ref_poses)
                    dy_ref_poses = np.sin(th_ref_poses)
                    plt.quiver(ref_poses[0, :], ref_poses[1, :], dx_ref_poses, dy_ref_poses, label='ref poses', color='b')
                    
                    future_poses = np.zeros((3, self.K))
                    prev_pose = curr_pose
                    for action_num in range(self.K):
                        action = applied_vels[action_num*2:action_num*2+2]
                        prev_pose = Unicycle.step_external(prev_pose, action, self.time_step)
                        future_poses[:, action_num] = prev_pose
            
                    th_future_poses = future_poses[2, :]
                    dx_future_poses = np.cos(th_future_poses)
                    dy_future_poses = np.sin(th_future_poses)
                    plt.quiver(future_poses[0, :], future_poses[1, :], dx_future_poses, dy_future_poses, label='future poses', color='y')

                    # set xlim
                    ax.set_xlim(-8, 8)
                    ax.set_ylim(-6, 6)
                    plt.legend()
                    ax.set_aspect('equal')
                    plt.savefig(f'mpc_debug_{curr_pose[0]}_{pose_error_scaling}.png')    

            except Exception as e:
                pose_error_scaling *= 5 # automatically tune for feasible solution
                continue
    
            # prep next iterations
            self.next_se3_state_vars = []
            self.next_vspace_state_vars = []

            for i in range(1, self.K, 1):
                prev_se3_state = se3_state_vars[i].value.matrix()
                prev_vspace_state = vspace_state_vars[i].value
                self.next_se3_state_vars.append(SE3StateVar(Transformation(T_ba=prev_se3_state)))
                self.next_vspace_state_vars.append(VSpaceStateVar(prev_vspace_state))

            # duplicate the last entry as an initial guess with zero end velocity
            prev_se3_state = se3_state_vars[self.K - 1].value.matrix()
            self.next_se3_state_vars.append(SE3StateVar(Transformation(T_ba=prev_se3_state)))
            self.next_vspace_state_vars.append(VSpaceStateVar(np.array([[0.5], [0]])))
        return applied_vels
    
    def generate_local_ref_path(self, path, interp_idx, se3_state_var=True):
            lower_idx = int(np.floor(interp_idx))
            upper_idx = int(np.ceil(interp_idx))

            ref_poses = []

            target_idx = interp_idx + self.node_idx_increment
            for i in range(self.K):
                if target_idx >= path.shape[1]-1:
                    ref_poses.append(path[:, -1])
                else:
                    lower_pose = path[:, int(np.floor(target_idx))]
                    upper_pose = path[:, int(np.ceil(target_idx))]
                    interp_pose = lower_pose + (target_idx - int(np.floor(target_idx))) * (upper_pose - lower_pose)
                    ref_poses.append(interp_pose)
                
                target_idx += self.node_idx_increment

            ref_SE3s = []
            for pose in ref_poses:
                T_ba_array = np.array([[np.cos(pose[2]), -np.sin(pose[2]), 0, pose[0]],
                                       [np.sin(pose[2]), np.cos(pose[2]), 0, pose[1]],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
                if se3_state_var:
                    ref_SE3s.append(SE3StateVar(Transformation(T_ba=T_ba_array).inverse()))
                else:
                    ref_SE3s.append(Transformation(T_ba=T_ba_array).inverse())

            return ref_SE3s, ref_poses

