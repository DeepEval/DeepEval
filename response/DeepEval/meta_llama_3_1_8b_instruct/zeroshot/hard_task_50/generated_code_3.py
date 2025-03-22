# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two blocks. The first block processes the input through three max pooling layers 
    with different scales, utilizing pooling windows and strides of 1x1, 2x2, and 4x4, respectively. 
    Each pooling output is flattened into a one-dimensional vector. To mitigate overfitting, dropout is 
    applied before concatenating these vectors. A fully connected layer and a reshape operation transform 
    the output of the first block into a four-dimensional tensor for processing in second block. The second 
    block splits the input into four groups along the last dimension by encapsulating tf.split within Lambda 
    layer. Each group is processed by a separable convolutional with varying kernel sizes (1x1, 3x3, 5x5, and 
    7x7) to extract features. The outputs from these four groups are concatenated. Finally, the model produces 
    the classification result by flattening the output and passing it through a fully connected layer.

    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """

    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the model architecture
    inputs = keras.Input(shape=input_shape)

    # First block
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=1)(inputs)
    x = layers.Flatten()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=4)(x)
    x = layers.Flatten()(x)

    # Apply dropout to mitigate overfitting
    x = layers.Dropout(0.2)(x)

    # Concatenate the outputs from three max pooling layers
    x = layers.Concatenate()([x, x, x])

    # Fully connected layer and reshape operation
    x = layers.Dense(128, activation=activations.relu)(x)
    x = layers.Reshape((4, 8, 2))(x)

    # Second block
    # Split the input into four groups along the last dimension
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(x)

    # Process each group by a separable convolutional with varying kernel sizes
    groups = []
    for i in range(4):
        group = layers.SeparableConv2D(32, (1, 1), activation=activations.relu)(x[i])
        group = layers.SeparableConv2D(32, (3, 3), activation=activations.relu)(group)
        group = layers.SeparableConv2D(32, (5, 5), activation=activations.relu)(group)
        group = layers.SeparableConv2D(32, (7, 7), activation=activations.relu)(group)
        groups.append(group)

    # Concatenate the outputs from four groups
    x = layers.Concatenate()(groups)

    # Flatten the output and pass it through a fully connected layer
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation=activations.softmax)(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model