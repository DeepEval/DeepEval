from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten, concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Path 2: Average pooling followed by 1x1 convolution
    path2 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path3_1 = Conv2D(32, (1, 3), activation='relu', padding='same')(path3)
    path3_2 = Conv2D(32, (3, 1), activation='relu', padding='same')(path3)
    path3 = concatenate([path3_1, path3_2], axis=-1)

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then two parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path4 = Conv2D(32, (3, 3), activation='relu', padding='same')(path4)
    path4_1 = Conv2D(32, (1, 3), activation='relu', padding='same')(path4)
    path4_2 = Conv2D(32, (3, 1), activation='relu', padding='same')(path4)
    path4 = concatenate([path4_1, path4_2], axis=-1)

    # Concatenate all paths
    concatenated = concatenate([path1, path2, path3, path4], axis=-1)

    # Flatten and add a fully connected layer for classification
    flat = Flatten()(concatenated)
    dense = Dense(10, activation='softmax')(flat)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=dense)

    return model

# Example usage
model = dl_model()
model.summary()