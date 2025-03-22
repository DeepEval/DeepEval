# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path consists of four components:
    Block 1, Block 2, Block 3, and Block 1 (repeated). The branch path extracts features from the original
    input using an average pooling layer. The outputs from both the main path and the branch path are then
    concatenated. Finally, the combined output is passed through a fully connected layer to complete the
    classification process.
    
    :return: The constructed model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define Block 1
    def block1(x):
        # Split the input into three groups
        groups = tf.split(x, 3, axis=-1)
        
        # Process each group using a 1x1 convolutional layer followed by an activation function
        conv1 = layers.Conv2D(16, kernel_size=1, activation='relu')(groups[0])
        conv2 = layers.Conv2D(16, kernel_size=1, activation='relu')(groups[1])
        conv3 = layers.Conv2D(16, kernel_size=1, activation='relu')(groups[2])
        
        # Concatenate the outputs from the three groups
        x = layers.Concatenate()([conv1, conv2, conv3])
        
        return x

    # Define Block 2
    def block2(x):
        # Get the shape of the features
        shape = tf.shape(x)
        
        # Reshape the feature into a target shape
        x = layers.Reshape((shape[1], shape[2], 3, 16))(x)
        
        # Swap the third and fourth dimensions
        x = layers.Lambda(lambda x: tf.transpose(x, (1, 2, 4, 0)))(x)
        
        # Reshape the feature back to its original shape
        x = layers.Reshape((shape[1], shape[2], 48))(x)
        
        return x

    # Define Block 3
    def block3(x):
        # Apply a 3x3 depthwise separable convolution to the output from Block 2
        x = layers.SeparableConv2D(32, kernel_size=3, activation='relu')(x)
        
        return x

    # Create the main path
    x = block1(inputs)
    x = block2(x)
    x = block3(x)
    x = block1(x)

    # Create the branch path
    x_branch = layers.AveragePooling2D(pool_size=8)(inputs)
    x_branch = layers.Flatten()(x_branch)

    # Concatenate the outputs from the main path and the branch path
    x = layers.Concatenate()([x, x_branch])

    # Create the fully connected layer
    x = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model