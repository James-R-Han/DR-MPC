import numpy as np

from pysteam.pysteam.evaluable.evaluable import Evaluable, Node, Jacobians

class LogEvaluator(Evaluable):
  """Evaluator for the log of a scalar."""

  def __init__(self, scalar: Evaluable) -> None:
    super().__init__()
    self._scalar: Evaluable = scalar

  @property
  def active(self) -> bool:
    return self._scalar.active

  @property
  def related_var_keys(self) -> set:
    return self._scalar.related_var_keys

  def forward(self) -> Node:
    child = self._scalar.forward()
    # print(child.value)

    # fake gradient to move between 0 and 1
    if child.value <= 0:
      value = np.array([[-500]])
    elif child.value > 1:
      value = np.array([[500]])
    else:
      value = np.log(child.value)
    # value = np.log(child.value)

    return Node(value, child)

  def backward(self, lhs: np.ndarray, node: Node, jacs: Jacobians) -> None:
    if self._scalar.active:
      # if node.children[0].value < 1e-5:
      #   print("hi")
      if node.children[0].value <= 0 :
        self._scalar.backward(lhs / 0.1, node.children[0], jacs)
      elif node.children[0].value > 1:
        self._scalar.backward(lhs / 0.1, node.children[0], jacs)
      else:
        self._scalar.backward(lhs/node.children[0].value, node.children[0], jacs)
