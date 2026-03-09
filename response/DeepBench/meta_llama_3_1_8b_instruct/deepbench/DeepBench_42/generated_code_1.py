import tensorflow as tf

class ArcCosine(Kernel):
    def __init__(self):
        super().__init__()
        self.weight_variances = 0.5
        self.bias_variance = 0.2

    def _full_weighted_product(self, X: tf.Tensor, X2: tf.Optional[tf.Tensor] = None) -> tf.Tensor:
        if X2 is None:
            return tf.multiply(X, tf.math.exp(-self.weight_variances * tf.square(X)) + self.bias_variance)
        else:
            return tf.multiply(X, X2)

if __name__ == "__main__":
    # Create sample input values
    tf.random.set_seed(42)
    X = tf.random.normal(shape=(3, 4), mean=0.0, stddev=1.0)
    X2 = tf.random.normal(shape=(3, 4), mean=0.0, stddev=1.0)

    # Create an instance of ArcCosine
    arc_cosine = ArcCosine()

    # Call the function and print the results
    print(arc_cosine._full_weighted_product(X))
    print(arc_cosine._full_weighted_product(X, X2))