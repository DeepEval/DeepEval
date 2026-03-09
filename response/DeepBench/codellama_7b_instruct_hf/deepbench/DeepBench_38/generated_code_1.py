import tensorflow as tf

def ndiag_mc(funcs, S, Fmu, Fvar, logspace=False, epsilon=None, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions using Monte Carlo samples. The Gaussians must be independent.

    :param funcs: the integrand(s): Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates the log-expectation of exp(funcs)
    :param Ys: arrays/tensors; deterministic arguments to be passed by name
    :return: shape is the same as that of the first Fmu
    """
    # Check input shapes
    @check_shapes("Fmu: [N, Din]", "Fvar: [N, Din]", "Ys.values(): [broadcast N, .]", "return: [broadcast n_funs, N, P]")
    def _check_shapes(Fmu, Fvar, Ys):
        # Check that all inputs have the same shape
        assert Fmu.shape == Fvar.shape
        assert all(Ys[i].shape[0] == Fmu.shape[0] for i in Ys)

    # Convert funcs to a list if it's not already
    if not isinstance(funcs, list):
        funcs = [funcs]

    # Get the number of functions and output dimensions
    n_funs = len(funcs)
    P = funcs[0].output_shape[1]

    # Get the input dimension
    Din = Fmu.shape[1]

    # Create a random Gaussian distribution for each function
    dists = [tfp.distributions.MultivariateNormalFull(Fmu, Fvar) for _ in range(n_funs)]

    # Create a placeholder for the Monte Carlo samples
    samples = tf.placeholder(tf.float32, shape=[None, Din])

    # Evaluate the functions at the Monte Carlo samples
    f_samples = [func(samples) for func in funcs]

    # Compute the log-integral of the functions
    log_integrals = tf.log(tf.reduce_sum(f_samples, axis=0))

    # Compute the expectation of the log-integrals
    log_expectation = tf.reduce_mean(log_integrals, axis=0)

    # Compute the Monte Carlo error of the expectation
    mc_error = tf.sqrt(tf.reduce_mean(tf.square(log_integrals - log_expectation), axis=0))

    # Compute the output
    output = tf.stack([log_expectation, mc_error], axis=0)

    # Create a session and run the function
    sess = tf.Session()
    output = sess.run(output, feed_dict={samples: Fmu.numpy()})

    # Close the session
    sess.close()

    return output

if __name__ == "__main__":
    # Create some sample inputs
    Fmu = tf.random.normal(shape=[100, 5])
    Fvar = tf.random.normal(shape=[100, 5])
    Ys = [tf.random.normal(shape=[100, 3])]

    # Call the function
    output = ndiag_mc(lambda x: tf.sin(x), 1000, Fmu, Fvar, **Ys)

    # Print the output
    print(output)