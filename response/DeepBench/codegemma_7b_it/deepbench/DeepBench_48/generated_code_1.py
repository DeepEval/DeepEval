import tensorflow as tf

def pyramid_combination(values, weight_floor, weight_ceil):
    ndim = len(weight_floor)
    if len(weight_ceil) != ndim:
        raise ValueError(f"weight_floor and weight_ceil must have the same dimensions, but have {ndim} and {len(weight_ceil)}.")
    for i in range(ndim):
        if values[i].shape != weight_floor[i].shape:
            raise ValueError(f"values[{i}] and weight_floor[{i}] must have the same shape.")
        if values[i].shape != weight_ceil[i].shape:
            raise ValueError(f"values[{i}] and weight_ceil[{i}] must have the same shape.")

    expected_n = 2 ** ndim
    if len(values) != expected_n:
        raise ValueError(f"values must have length {expected_n}, but have {len(values)}.")

    return tf.reduce_sum([
        values[i] * weight_floor[i] for i in range(ndim)
    ], axis=0) + tf.reduce_sum([
        values[expected_n // 2 + i] * weight_ceil[i] for i in range(ndim)
    ], axis=0)

if __name__ == "__main__":
    # Example usage:
    values = [
        tf.constant([0.0, 0.0]),
        tf.constant([0.0, 1.0]),
        tf.constant([1.0, 0.0]),
        tf.constant([1.0, 1.0]),
    ]
    weight_floor = [
        tf.constant([0.25, 0.25]),
        tf.constant([0.75, 0.25]),
    ]
    weight_ceil = [
        tf.constant([0.25, 0.75]),
        tf.constant([0.75, 0.75]),
    ]

    result = pyramid_combination(values, weight_floor, weight_ceil)
    print(result)