import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Path 2: Average pooling followed by 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
    path3 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    conv1x3 = Conv2D(32, (1, 3), activation='relu', padding='same')(path3)
    conv3x1 = Conv2D(32, (3, 1), activation='relu', padding='same')(path3)
    path3 = Concatenate()([conv1x3, conv3x1])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
    path4 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path4 = Conv2D(32, (3, 3), activation='relu', padding='same')(path4)
    conv1x3_path4 = Conv2D(32, (1, 3), activation='relu', padding='same')(path4)
    conv3x1_path4 = Conv2D(32, (3, 1), activation='relu', padding='same')(path4)
    path4 = Concatenate()([conv1x3_path4, conv3x1_path4])

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten and fully connected layer for classification
    flatten = Flatten()(concatenated)
    output_layer = Dense(10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Print the model summary to check the architecture