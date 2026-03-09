import tensorflow as tf

def canny(input_image, low_threshold, high_threshold, kernel_size, sigma, hysteresis, eps):
    # Check input image shape
    if len(input_image.shape) != 4 or input_image.shape[1] != 3:
        raise ValueError("Input image must be a 4D tensor with 3 color channels")

    # Convert input image to grayscale
    gray_image = tf.image.rgb_to_grayscale(input_image)

    # Apply Gaussian blur
    blurred_image = tf.nn.gaussian_filter(gray_image, kernel_size, sigma)

    # Compute gradients
    gradients = tf.image.sobel(blurred_image)

    # Compute gradient magnitude and angle
    gradient_magnitude = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=-1))
    gradient_angle = tf.atan2(gradients[..., 0], gradients[..., 1])

    # Non-maximal suppression
    suppression_mask = tf.where(gradient_magnitude < gradient_magnitude[..., 0], 0, 1)
    suppressed_gradient_magnitude = gradient_magnitude * suppression_mask

    # Apply thresholding
    threshold_mask = tf.where(suppressed_gradient_magnitude < low_threshold, 0, 1)
    thresholded_gradient_magnitude = suppressed_gradient_magnitude * threshold_mask

    # Apply hysteresis
    hysteresis_mask = tf.where(thresholded_gradient_magnitude > high_threshold, 1, 0)
    hysteresis_gradient_magnitude = thresholded_gradient_magnitude * hysteresis_mask

    # Return edge magnitudes and edge detection filtered by thresholds and hysteresis
    return gradient_magnitude, hysteresis_gradient_magnitude

if __name__ == "__main__":
    # Create a sample input image
    input_image = tf.random.normal(shape=[1, 3, 224, 224])

    # Call the function
    edge_magnitudes, edge_detection = canny(input_image, low_threshold=10, high_threshold=100, kernel_size=5, sigma=1, hysteresis=True, eps=0.01)

    # Print the edge magnitudes and edge detection
    print(edge_magnitudes)
    print(edge_detection)