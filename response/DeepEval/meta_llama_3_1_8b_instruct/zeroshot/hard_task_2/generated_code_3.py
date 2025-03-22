# Import necessary packages
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model splits the input into three groups along the channel dimension by encapsulating tf.split within a Lambda layer.
    Each group passes through a series of convolutions: a 1x1 convolution, followed by a 3x3 convolution, and another 1x1 convolution to extract features.
    The outputs from these three groups are combined using an addition operation to form the main path.
    Next, the main path is fused with the original input layer through another addition.
    Finally, the combined features are flattened into a one-dimensional vector, which is fed into a fully connected layer for classification.
    
    Returns:
        model (Model): The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    x = Lambda(lambda z: tf.split(z, 3, axis=-1))(inputs)

    # Define the first group of convolutions
    group1 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x[0])
    group1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group1)
    group1 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group1)

    # Define the second group of convolutions
    group2 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x[1])
    group2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group2)
    group2 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group2)

    # Define the third group of convolutions
    group3 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(x[2])
    group3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group3)
    group3 = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01))(group3)

    # Combine the outputs from the three groups using an addition operation
    x = Add()([group1, group2, group3])

    # Fuse the main path with the original input layer through another addition
    x = Add()([x, inputs])

    # Flatten the combined features into a one-dimensional vector
    x = Flatten()(x)

    # Define the fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()