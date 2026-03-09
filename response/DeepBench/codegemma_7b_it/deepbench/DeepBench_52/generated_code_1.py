import tensorflow as tf

def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """Create a 3d separable filter."""
    return tf.nn.conv3d(tensor, (kernel, tf.reduce_sum(kernel)), (1, 1, 1, 1, 1))

if __name__ == "__main__":
    # Create sample input values
    input_tensor = tf.constant(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ],
            [
                [9, 10, 11],
                [12, 13, 14],
                [15, 16, 17],
            ],
        ],
        dtype=tf.float32,
    )
    kernel = tf.constant([0.25, 0.5, 0.25], dtype=tf.float32)

    # Call the function and print the results
    output = separable_filter(input_tensor, kernel)
    print(output)