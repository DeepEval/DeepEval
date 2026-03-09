import tensorflow as tf
from typing import Optional, Any

class ArcCosine(object):
    def __init__(self, weight_variances: float, bias_variance: float):
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance

    def _full_weighted_product(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        if X2 is None:
            X2 = X
        weighted_product = tf.tensordot(X, X2, [[1], [1]]) * self.weight_variances + self.bias_variance
        return weighted_product

if __name__ == "__main__":
    # Example usage
    kernel = ArcCosine(weight_variances=2.0, bias_variance=1.0)
    X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    X2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    result = kernel._full_weighted_product(X, X2)
    print(result)  # Expected output: weighted product of X and X2 with given variances