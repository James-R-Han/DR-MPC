from pysteam.pysteam.problem import LossFunc

class L2LossFuncPose(LossFunc):

  def cost(self, whitened_error_norm: float) -> float:
    return 0.5 * whitened_error_norm * whitened_error_norm

  def weight(self, whitened_error_norm: float):
    return 3.0