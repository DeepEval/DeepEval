from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import activations

def dl_model():
    """
    Function to create a deep learning model for image classification using the
  MNIST dataset.
    
    The model is structured into two blocks. The first block processes the input
  through 
    three average pooling layers with varying scales, utilizing pooling windows
  and strides of 1x1, 2x2, and 4x4.
    Each pooling result is flattened into a one-dimensional vector, and these vectors
  are concatenated.
    Between the first and second blocks, a fully connected layer and a reshape
  operation transform the output of the first block 
    into a 4-dimensional tensor, suitable for input into the second block.
    The second block splits the input into four groups along the last dimension
  by encapsulating tf.split within Lambda layer, 
    each processed by depthwise separable convolutional layers with different kernel
  sizes (1x1, 3x3, 5x5, and 7x7) for feature extraction.
    The outputs from these groups are then concatenated. Finally, the processed
  data is flattened and passed through a fully connected layer 
    to produce the classification result.
    
    Returns:
    model: The constructed Keras model.
    """
    
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Create the model
    model = models.Sequential()
    
    # Block 1: Average Pooling Layers
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Average pooling layer with 1x1 window and stride of 1x1
    model.add(layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding=same))
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Average pooling layer with 2x2 window and stride of 2x2
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding=same))
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Average pooling layer with 4x4 window and stride of 4x4
    model.add(layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding=same))
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Concatenate the flattened vectors
    model.add(layers.Concatenate())
    
    # Block 2: Fully Connected and Reshape Layers
    # Fully connected layer with 128 units
    model.add(layers.Dense(128, activation=relu))
    
    # Reshape the output into a 4-dimensional tensor
    model.add(layers.Reshape((4, 16)))
    
    # Lambda layer to split the input into four groups
    model.add(layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1)))
    
    # Depthwise separable convolutional layers with different kernel sizes
    model.add(layers.SeparableConv2D(32, (1, 1), activation=relu, padding=same))
    model.add(layers.SeparableConv2D(64, (3, 3), activation=relu, padding=same))
    model.add(layers.SeparableConv2D(128, (5, 5), activation=relu, padding=same))
    model.add(layers.SeparableConv2D(256, (7, 7), activation=relu, padding=same))
    
    # Concatenate the outputs from the depthwise separable convolutional layers
    model.add(layers.Concatenate())
    
    # Flatten the output
    model.add(layers.Flatten())
    
    # Fully connected layer for classification
    model.add(layers.Dense(10, activation=softmax))
    
    # Compile the model
    model.compile(optimizer=adam,
                  loss=sparse_categorical_crossentropy,
                  metrics=[accuracy])
    
    return model