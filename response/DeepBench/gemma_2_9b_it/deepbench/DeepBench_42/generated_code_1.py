import tensorflow as tf

class ArcCosine(Kernel):
    def _full_weighted_product(self, X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
        if X2 is None:
            return tf.matmul(X, X) * self.weight_variances
        return tf.matmul(X, X2) * self.weight_variances + tf.matmul(X2, X) * self.bias_variance

if __name__ == "__main__":
    X = tf.random.normal((3, 4))
    X2 = tf.random.normal((4, 5))
    arc_cosine = ArcCosine()
    arc_cosine.weight_variances = tf.random.uniform([])
    arc_cosine.bias_variance = tf.random.uniform([])
    result = arc_cosine._full_weighted_product(X, X2)
    print(result)