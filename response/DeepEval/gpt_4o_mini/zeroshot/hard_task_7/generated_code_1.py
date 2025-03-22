import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for MNIST dataset (28x28 images with 1 channel)
    input_layer = Input(shape=(28, 28, 1))

    # Initial convolutional layer with 32 kernels
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Split the output into two groups along the last dimension
    def split_and_process(tensor):
        group1, group2 = tf.split(tensor, num_or_size_splits=2, axis=-1)

        # Group 1: Series of operations
        group1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(group1)
        group1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group1)
        group1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(group1)

        # Return concatenated outputs
        return Concatenate(axis=-1)([group1, group2])

    # Apply the Lambda layer to split and process the input
    x = Lambda(split_and_process)(x)

    # Block 2: Reshape the input into four groups
    def reshape_and_permute(tensor):
        # Get the shape of the input tensor
        shape = tf.shape(tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        
        # Reshape into groups
        groups = 4
        channels_per_group = channels // groups
        tensor = tf.reshape(tensor, (shape[0], height, width, groups, channels_per_group))

        # Permute the tensor to shuffle channels
        tensor = tf.transpose(tensor, perm=[0, 1, 2, 4, 3])

        # Reshape back to original shape
        return tf.reshape(tensor, (shape[0], height, width, channels))

    # Apply the Lambda layer for reshaping and permutation
    x = Lambda(reshape_and_permute)(x)

    # Flatten and add a fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()