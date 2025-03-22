import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(64, (1, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    path3a = Conv2D(64, (1, 3), activation='relu', padding='same')(path3)
    path3b = Conv2D(64, (3, 1), activation='relu', padding='same')(path3)
    path3 = Concatenate()([path3a, path3b])

    # Path 4: 1x1 convolution, 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    path4 = Conv2D(64, (3, 3), activation='relu', padding='same')(path4)
    path4a = Conv2D(64, (1, 3), activation='relu', padding='same')(path4)
    path4b = Conv2D(64, (3, 1), activation='relu', padding='same')(path4)
    path4 = Concatenate()([path4a, path4b])

    # Concatenate all paths
    merged = Concatenate()([path1, path2, path3, path4])

    # Flatten and add fully connected layer for classification
    flatten = Flatten()(merged)
    dense = Dense(10, activation='softmax')(flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model

# Example usage
model = dl_model()
model.summary()