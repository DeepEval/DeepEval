import tensorflow as tf
from typing import Callable, Optional, Tuple

def check_shapes(
    means: str,
    covs: str,
    return_: str,
):
    """ Decorator for checking the shapes of the input and output tensors. """

    def decorator(func: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
        def wrapper(*args, **kwargs) -> tf.Tensor:
            # Check the shape of the input tensors.
            for key, shape in zip(
                ("means", "covs"),
                (tf.TensorShape(means.split(":")[1].strip()), tf.TensorShape(covs.split(":")[1].strip())),
            ):
                if not tf.TensorShape(args[list(kwargs).index(key)]).is_compatible_with(shape):
                    raise ValueError(f"Invalid shape for {key}. Expected {shape}, but got {tf.TensorShape(args[list(kwargs).index(key)])}.")

            # Call the function and return the output tensor.
            return func(*args, **kwargs)

        return wrapper

    return decorator

@check_shapes(
    means: "N, Din",
    covs: "N, Din, Din",
    return_: "N, Dout...",
)
def mvnquad(
    func: Callable[[tf.Tensor], tf.Tensor],
    means: tf.Tensor,
    covs: tf.Tensor,
    H: int,
    Din: Optional[int] = None,
    Dout: Optional[Tuple[int, ...]] = None,
) -> tf.Tensor:
    """ Computes N Gaussian expectation integrals of a single function 'f' using Gauss-Hermite quadrature. :param f: integrand function. Takes one input of shape ?xD. :param H: Number of Gauss-Hermite evaluation points. :param Din: Number of input dimensions. Needs to be known at call-time. :param Dout: Number of output dimensions. Defaults to (). Dout is assumed to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout). :return: quadratures """

    # Figure out input shape information if Din is None.
    if Din is None:
        Din = means.shape[1]
    if Dout is None:
        Dout = tuple()

    # Generate Gauss-Hermite quadrature points and weights.
    quad_points, quad_weights = tf.linalg.gaussian_quadrature(H, Dout)

    # Evaluate the integrand function at the quadrature points.
    quad_values = func(tf.einsum("nki,kh->nkh", means, quad_points))

    # Calculate the expectation integrals.
    quadratures = tf.einsum("nkh,nk->nh", quad_values, quad_weights)
    quadratures = tf.einsum("nh,n->h", quadratures, tf.reduce_sum(quad_weights, axis=1))

    # Reshape the output tensors.
    return tf.reshape(quadratures, [-1] + list(Dout))

if __name__ == "__main__":
    # Create sample input values.
    means = tf.random.normal([2, 5])
    covs = tf.eye(5, batch_shape=[2])
    H = 50

    # Define the integrand function.
    def integrand(x):
        return tf.math.sin(x)

    # Call the mvnquad function.
    quadratures = mvnquad(integrand, means, covs, H)

    # Print the results.
    print(quadratures)