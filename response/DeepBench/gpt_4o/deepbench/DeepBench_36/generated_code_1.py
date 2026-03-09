import numpy as np
from typing import Any

TensorType = Any

class Kernel:
    pass

class ArcCosine(Kernel):
    def __init__(self, order: int):
        self.order = order

    def _J(self, theta: TensorType) -> TensorType:
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return np.sin(theta) + (np.pi - theta) * np.cos(theta)
        elif self.order == 2:
            return 3 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (1 + 2 * np.cos(theta)**2)
        else:
            raise ValueError("Order not supported. Please use order 0, 1, or 2.")

if __name__ == "__main__":
    theta = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    
    arc_cosine_0 = ArcCosine(order=0)
    result_0 = arc_cosine_0._J(theta)
    print("Order 0:", result_0)
    
    arc_cosine_1 = ArcCosine(order=1)
    result_1 = arc_cosine_1._J(theta)
    print("Order 1:", result_1)
    
    arc_cosine_2 = ArcCosine(order=2)
    result_2 = arc_cosine_2._J(theta)
    print("Order 2:", result_2)