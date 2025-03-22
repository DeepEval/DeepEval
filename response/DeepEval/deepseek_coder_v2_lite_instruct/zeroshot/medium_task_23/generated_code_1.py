import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the inputs
    inputs = Input(shape=input_shape)

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(32, (1, 1), activation='relu')(inputs)
    path3 = Conv2D(32, (1, 7), activation='relu', padding='same')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu', padding='same')(path3)
    path3 = Conv2D(32, (1, 7), activation='relu', padding='same')(path3)
    path3 = Conv2D(32, (7, 1), activation='relu', padding='same')(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D((8, 8))(inputs)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate the outputs of the four paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()