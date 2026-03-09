import tensorflow as tf

class ArcCosine(Kernel):
    def __init__(self, weight_variances, bias_variance):
        self.weight_variances = weight_variances
        self.bias_variance = bias_variance

    def _full_weighted_product(self, X: TensorType, X2: Optional[TensorType]) -> tf.Tensor:
        if X2 is None:
            X2 = X
        return tf.matmul(X, X2) * self.weight_variances + self.bias_variance

if __name__ == "__main__":
    # Create sample input values
    X = tf.random.normal([10, 5])
    X2 = tf.random.normal([10, 5])

    # Call the function and print the results
    kernel = ArcCosine(weight_variances=0.5, bias_variance=0.1)
    result = kernel._full_weighted_product(X, X2)
    print(result)