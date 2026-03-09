import tensorflow as tf

def _full_weighted_product(self, X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    if X2 is None:
        return tf.matmul(tf.sqrt(self.weight_variances), X) + tf.sqrt(self.bias_variance)
    else:
        return tf.matmul(tf.sqrt(self.weight_variances), X) + tf.matmul(tf.sqrt(self.weight_variances), X2) + tf.sqrt(self.bias_variance)

if __name__ == "__main__":
    # Create sample input values
    X = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    X2 = tf.constant([[5.0, 6.0], [7.0, 8.0]])

    # Set weight and bias variances
    weight_variances = tf.constant([0.1, 0.2])
    bias_variance = tf.constant(0.3)

    # Create an instance of ArcCosine class
    arccosine = ArcCosine()

    # Compute weighted product using _full_weighted_product function
    weighted_product = arccosine._full_weighted_product(X, X2)

    # Print the results
    print(weighted_product)