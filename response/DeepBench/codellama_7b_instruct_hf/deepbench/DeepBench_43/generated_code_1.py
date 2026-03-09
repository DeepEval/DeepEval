import tensorflow as tf

@check_shapes(("x", [D, None]), ("mu", [D, None]), ("L", [D, D]))
def multivariate_normal(x: TensorType, mu: TensorType, L: TensorType) -> tf.Tensor:
    # Compute the log-density of the multivariate normal
    log_density = tf.reduce_sum(
        -0.5 * tf.matmul(
            tf.linalg.matrix_transpose(x - mu),
            tf.linalg.matmul(L, x - mu)
        ),
        axis=1
    )
    return log_density

if __name__ == "__main__":
    # Create some sample input values
    x = tf.random.normal([4, 2])
    mu = tf.random.normal([4, 2])
    L = tf.random.normal([4, 2, 2])

    # Call the function and print the results
    print(multivariate_normal(x, mu, L))