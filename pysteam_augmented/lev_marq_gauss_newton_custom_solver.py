import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp_sparse

from pysteam.pysteam.problem import Problem
from pysteam.pysteam.solver.gauss_newton_solver import GaussNewtonSolver
from pysteam.pysteam.evaluable.vspace import VSpaceStateVar


class LevMarqGaussNewtonCustomSolver(GaussNewtonSolver):

  def __init__(self, problem: Problem, **parameters) -> None:
    super().__init__(problem, **parameters)
    # override parameters
    self._parameters.update({
        "ratio_threshold": 0.25,
        "shrink_coeff": 0.1,
        "grow_coeff": 10.0,
        "max_shrink_steps": 50,
        'backtrack_multiplier': 0.8,
        'max_backtrack_steps': 2,
    })
    self._parameters.update(**parameters)
    self.v_max = parameters['v_max']
    self.v_min = parameters['v_min']
    self.w_max = parameters['w_max']
    self.w_min = parameters['w_min']
    self._diag_coeff = 1e-7

  def linearize_solve_and_update(self):

    # initialize new cost with old cost in case of failure
    new_cost = self._prev_cost

    # build the system
    A, b = self._problem.build_gauss_newton_terms()
    grad_norm = npla.norm(b)  # compute gradient norm for termination check

    # perform LM search
    num_tr_decreases = 0
    num_backtrack = 0
    step_success = False
    while num_backtrack < self._parameters["max_shrink_steps"]:
      try:
        perturbation = self.solve_lev_marq(A, b)
        decomp_success = True
      except npla.LinAlgError:
        decomp_success = False

      if decomp_success:
        # JH: I changed this to backtrack to satisify log constraints on the action space
        max_valid_perturbation_scaling = 1

        for state in self._state_vector._state_vars.values():
          if state.state_var.locked:
            continue
          if isinstance(state.state_var, VSpaceStateVar):
            curr_value = state.state_var.value
            v = curr_value[0, 0]
            w = curr_value[1, 0]
            relevant_perturbation = perturbation[state.indices]
            # assuming v and w in right range (ie, the backtrack range)
            v_perturb = relevant_perturbation[0, 0]
            w_perturb = relevant_perturbation[1, 0]
            if v_perturb != 0:
              if v_perturb > 0:
                max_v_perturb = (self.v_max - 1e-7) - v
              else:
                # max_v_perturb = v - (0 + 1e-15)
                  # want negative since we are dividing by a negative
                  max_v_perturb = (self.v_min + 1e-7) - v
              allowed_perturbation_percentage = max_v_perturb / v_perturb
              max_valid_perturbation_scaling = min(max_valid_perturbation_scaling, allowed_perturbation_percentage)
              if allowed_perturbation_percentage < 1:
                max_valid_perturbation_scaling = max(max_valid_perturbation_scaling-1e-7, 0)

              if max_valid_perturbation_scaling > 1 or max_valid_perturbation_scaling < 0:
                print("what1")
            if w_perturb != 0:
              if w_perturb > 0:
                max_w_perturb = (self.w_max - 1e-7) - w
              else:
                # max_w_perturb = w - (-1 + 1e-15)
                # want negative since we are dividing by a negative
                max_w_perturb = (self.w_min + 1e-7) - w
              allowed_perturbation_percentage = max_w_perturb / w_perturb
              max_valid_perturbation_scaling = min(max_valid_perturbation_scaling, allowed_perturbation_percentage)
              if allowed_perturbation_percentage < 1:
                max_valid_perturbation_scaling = max(max_valid_perturbation_scaling-1e-7, 0)
              if max_valid_perturbation_scaling > 1 or max_valid_perturbation_scaling < 0:
                print("what2")

        # Having trouble satisifing trust region. Perform line search to get reduce perturbation.
        break_loop = False
        perturbation = perturbation * max_valid_perturbation_scaling
        for backtrack_step in range(self._parameters["max_backtrack_steps"]):
          if backtrack_step > 0:
            perturbation = perturbation * self._parameters["backtrack_multiplier"]
          proposed_cost = self.propose_update(perturbation)
          actual_reduc = self._prev_cost - proposed_cost
          predicted_reduc = self.predict_reduction(A, b, perturbation)
          if predicted_reduc == 0:
            actual_to_predicted_ratio = 0.0
          else:
            try:
              actual_to_predicted_ratio = actual_reduc / predicted_reduc
            # except everything and print it out
            except Exception as e:
              print(e)
              print("actual_reduc", actual_reduc)
              print("predicted_reduc", predicted_reduc)
              actual_to_predicted_ratio = 0.0

          if decomp_success and actual_to_predicted_ratio > self._parameters["ratio_threshold"]:
            self.accept_proposed_state()
            self._diag_coeff = max(self._diag_coeff * self._parameters["shrink_coeff"], 1e-7)
            new_cost = proposed_cost
            step_success = True
            break_loop = True
            break
          else:
            self.reject_proposed_state()
        if break_loop:
          break
        else:
          self._diag_coeff = min(self._diag_coeff * self._parameters["grow_coeff"], 1e7)
          num_tr_decreases += 1
        
      else:
        actual_to_predicted_ratio = 0.0
        self._diag_coeff = min(self._diag_coeff * self._parameters["grow_coeff"], 1e7)
        num_tr_decreases += 1

      num_backtrack += 1

    # print report line if verbose option is enabled
    if (self._parameters["verbose"]):
      print(f"Iteration: {self._curr_iteration:4}  -  Cost: {new_cost:10.4f}  -  TR Shrink: {num_tr_decreases:6.3f}  -  AvP Ratio: {actual_to_predicted_ratio:6.3f}")
    # if step_success is False:
    #   print("step failed")
    return step_success, new_cost, grad_norm

  def solve_lev_marq(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the Levenberg–Marquardt system of equations:
      A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
    """
    # augment diagonal of the 'hessian' matrix
    if sp_sparse.issparse(A):
      A.setdiag(A.diagonal() * (1 + self._diag_coeff))
    else:
      np.fill_diagonal(A, np.diag(A) * (1 + self._diag_coeff))

    # solve system
    try:
      lev_marq_step = self.solve_gauss_newton(A, b)
    except npla.LinAlgError:
      raise npla.LinAlgError('Decomposition Failure')
    finally:
      # revert diagonal of the 'hessian' matrix
      if sp_sparse.issparse(A):
        A.setdiag(A.diagonal() / (1 + self._diag_coeff))
      else:
        np.fill_diagonal(A, np.diag(A) / (1 + self._diag_coeff))

    return lev_marq_step

  def predict_reduction(self, A: np.ndarray, b: np.ndarray, step: np.ndarray) -> float:
    # grad^T * step - 0.5 * step^T * Hessian * step
    grad_trans_step = b.T @ step
    step_trans_hessian_step = step.T @ A @ step
    return (grad_trans_step - 0.5 * step_trans_hessian_step)[0, 0]
