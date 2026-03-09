import tensorflow as tf
import numpy as np

@tf.function
@tf.custom_gradient
def multivariate_normal(x: tf.Tensor, mu: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    @tf.function
    def grad(dlogp, var):
        N = tf.shape(x)[0]
        D = tf.shape(x)[1]
        if var == mu:
            return -tf.matmul(tf.transpose(tf.eye(D) - tf.linalg.matmul(L, L, transpose_b=True)), dlogp)
        elif var == L:
            return tf.matmul(tf.matmul(L, tf.transpose(x - mu, [0, 2, 1])), dlogp) + tf.matmul(tf.transpose(tf.matmul(L, tf.transpose(x - mu, [0, 2, 1])), [0, 2, 1]), L)

    D = tf.shape(x)[1]
    N = tf.shape(x)[0]
    logdet_L = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(L)))
    x_centered = x - mu
    det_L = tf.linalg.det(L)
    norm_term = -0.5 * tf.cast(N * D, tf.float64) * np.log(2 * np.pi)
    log_density = norm_term - 0.5 * tf.reduce_sum(tf.matmul(tf.transpose(x_centered, [0, 2, 1]), tf.linalg.solve(L, x_centered)), axis=[1, 2]) - 0.5 * logdet_L
    grad_mu = lambda: -tf.matmul(tf.transpose(tf.eye(D) - tf.linalg.matmul(L, L, transpose_b=True)), tf.reshape(dlogp, [N, D]))
    grad_L = lambda: tf.matmul(tf.matmul(L, tf.transpose(x - mu, [0, 2, 1])), tf.reshape(dlogp, [N, D])) + tf.matmul(tf.transpose(tf.matmul(L, tf.transpose(x - mu, [0, 2, 1])), [0, 2, 1]), L)
    
    return tf.reduce_sum(log_density, axis=1), (grad_mu, grad_L)

if __name__ == "__main__":
    # Create sample input values
    N = 10
    D = 5
    x = tf.random.normal([N, D])
    mu = tf.random.normal([D])
    L = tf.linalg.cholesky(tf.random.normal([D, D]))
    
    # Call the function and print the results
    log_density, _ = multivariate_normal(x, mu, L)
    print("Log density:", log_density)