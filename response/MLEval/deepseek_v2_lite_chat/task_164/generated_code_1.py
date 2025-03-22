import numpy as np

def squashing_function(vectors):
    """
    Applies the squashing function to vectors.
    Args:
    vectors (array): A 2D array of vectors to be squashed.
    Returns:
    array: A 2D array with the squashed vectors.
    """
    squared_norms = np.sum(np.square(vectors), axis=1)
    result = squared_norms * (1.0 + np.sqrt(np.sum(np.square(vectors), axis=1)))
    return vectors * result[:, None]

def capsule_layer(input_vectors, class_length, num_filters, class_mask=None):
    """
    Performs a capsule layer transformation.
    Args:
    input_vectors (array): A 2D array of input vectors.
    class_length (int): The dimension of each class vector.
    num_filters (int): The number of filters in the capsule layer.
    class_mask (array, optional): A mask for selecting specific classes.
    Returns:
    array: A 2D array with the transformed capsule vectors.
    """
    # Reshape the input to accommodate the capsule layer
    batch_size, input_length, _ = input_vectors.shape
    flattened_vectors = input_vectors.reshape(batch_size, input_length * num_filters)
    
    # Initialize transformed vectors
    output_vectors = np.zeros((batch_size, num_filters, class_length, class_length))
    
    # Process each class
    for i in range(class_length):
        for j in range(num_filters):
            class_vectors = flattened_vectors[np.nonzero(class_mask == i)]
            class_output = squashing_function(class_vectors.reshape(class_vectors.shape[0], class_vectors.shape[1], 1, 1))
            output_vectors[np.nonzero(class_mask == i)] = class_output.reshape(output_vectors.shape[1:]) * class_mask[np.nonzero(class_mask == i)][:, None, None, None]
    
    return output_vectors

# Example usage:
# Create some random input data and a simple mask
input_vectors = np.random.rand(2, 5, 10)  # (batch_size, input_length, features)
class_mask = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 1, 0]])  # (batch_size, class_length)

# Call the function with the input data and mask
output = capsule_layer(input_vectors, class_length=3, num_filters=4, class_mask=class_mask)
print(output.shape)  # Should output (2, 4, 3, 3)