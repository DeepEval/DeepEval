import tensorflow as tf

def multivariate_normal(x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """
    Computes the log-density of a multivariate normal.

    :param x: sample(s) for which we want the density
    :param mu: mean(s) of the normal distribution
    :param L: Cholesky decomposition of the covariance matrix
    :return: log densities
    """

    # Reshape input tensors
    x = tf.reshape(x, [-1, tf.shape(x)[-1]])
    mu = tf.reshape(mu, [-1, tf.shape(mu)[-1]])
    L = tf.reshape(L, [-1, tf.shape(L)[0], tf.shape(L)[1]])

    # Cholesky decomposition of the covariance matrix
    L_chol = tf.linalg.cholesky(L)

    # Calculate the log-density
    log_density = -0.5 * tf.reduce_sum(tf.square((x - mu) / tf.linalg.cholesky(L_chol)), axis=-1) - 0.5 * tf.reduce_sum(tf.log(tf.linalg.diag(tf.linalg.cholesky(L))), axis=-1) - 0.5 * tf.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32)

    return log_density

if __name__ == "__main__":
    # Create sample input values
    x = tf.constant([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    mu = tf.constant([0.0, 0.0])
    L = tf.constant([[1.0, 0.0], [0.0, 1.0]])

    # Call the function and print the results
    log_density = multivariate_normal(x, mu, L)
    print(log_density)