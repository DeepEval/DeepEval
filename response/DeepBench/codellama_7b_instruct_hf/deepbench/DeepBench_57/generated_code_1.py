import tensorflow as tf

def mvnquad(func, means, covs, H, Din=None, Dout=None):
    # Check input shapes
    @tf.function
    def check_shapes(means, covs, Din, Dout):
        # Check input shapes
        if Din is None:
            Din = means.shape[1]
        if Dout is None:
            Dout = ()
        # Compute output shape
        Dout = (Din - len(Dout),) + Dout

    # Evaluate function at Gauss-Hermite points
    @tf.function
    def evaluate_function(func, means, covs, H, Din, Dout):
        # Compute Gauss-Hermite points and weights
        points, weights = tf.nn.gauss_hermite_pts_wts(H)
        points = points[:, :Din]
        weights = weights[:, None]
        # Evaluate function at Gauss-Hermite points
        func_vals = func(points)
        # Compute quadrature
        quadrature = tf.matmul(weights, tf.matmul(covs, func_vals))
        return quadrature

    # Compute N Gaussian expectation integrals
    @tf.function
    def compute_expectations(means, covs, H, Din, Dout):
        # Compute number of Gaussian expectations
        N = means.shape[0]
        # Evaluate function at Gauss-Hermite points
        func_vals = evaluate_function(func, means, covs, H, Din, Dout)
        # Compute Gaussian expectations
        expectations = tf.reshape(func_vals, (N, -1))
        return expectations

    # Compute N Gaussian expectation integrals of a single function 'f'
    # using Gauss-Hermite quadrature.
    @tf.function
    def mvnquad(func, means, covs, H, Din=None, Dout=None):
        # Check input shapes
        check_shapes(means, covs, Din, Dout)
        # Evaluate function at Gauss-Hermite points
        expectations = compute_expectations(means, covs, H, Din, Dout)
        return expectations

if __name__ == "__main__":
    # Sample input values
    means = tf.random.normal(shape=(10, 2))
    covs = tf.random.normal(shape=(10, 2, 2))
    H = 20
    Din = 2
    Dout = ()

    # Call function and print results
    results = mvnquad(means, covs, H, Din, Dout)
    print(results)