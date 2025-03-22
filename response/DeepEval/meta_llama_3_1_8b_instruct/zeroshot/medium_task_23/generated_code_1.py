# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def dl_model():
    """
    This function generates a deep learning model for image classification using the CIFAR-10 dataset.
    The model employs a multi-branch convolutional structure.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Path 1: A single 1x1 convolution
    x1 = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
    x1 = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x1)

    # Path 2: A 1x1 convolution followed by 1x7 and 7x1 convolutions
    x2 = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
    x2 = Conv2D(32, kernel_size=(1, 7), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x2)
    x2 = Conv2D(32, kernel_size=(7, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x2)

    # Path 3: A 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions
    x3 = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(inputs)
    x3_1 = Conv2D(32, kernel_size=(1, 7), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x3)
    x3_1 = Conv2D(32, kernel_size=(7, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x3_1)
    x3_2 = Conv2D(32, kernel_size=(1, 7), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x3)
    x3_2 = Conv2D(32, kernel_size=(7, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x3_2)
    x3 = Concatenate()([x3_1, x3_2])

    # Path 4: Average pooling followed by a 1x1 convolution
    x4 = AveragePooling2D(pool_size=(2, 2))(inputs)
    x4 = Conv2D(32, kernel_size=(1, 1), activation='relu', kernel_initializer=glorot_uniform(seed=0))(x4)

    # Concatenate the outputs of the paths
    x = Concatenate()([x1, x2, x3, x4])

    # Flatten the concatenated output
    x = Flatten()(x)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model