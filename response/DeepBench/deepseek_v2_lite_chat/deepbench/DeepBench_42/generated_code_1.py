import tensorflow as tf

def _full_weighted_product(X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    if X2 is None:
        return tf.multiply(X, X)
    else:
        return tf.multiply(X, X2)

if __name__ == "__main__":
    # Sample input values
    X = tf.constant([1, 2, 3])
    X2 = tf.constant([4, 5, 6])
    
    # Call the function and print the results
    result = _full_weighted_product(X, X2)
    print("Full Weighted Product:", result.numpy())