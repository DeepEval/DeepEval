import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels
    
    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Path 2: 1x1 convolution followed by two 3x3 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (3, 3), activation='relu', padding='same')(path2)

    # Path 3: 1x1 convolution followed by a single 3x3 convolution
    path3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3 = Conv2D(32, (3, 3), activation='relu', padding='same')(path3)

    # Path 4: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = Conv2D(32, (1, 1), activation='relu')(path4)

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the output
    flatten = Flatten()(concatenated)

    # Dense layer with 128 units
    dense_layer = Dense(128, activation='relu')(flatten)

    # Output layer with softmax activation for 10 classes
    output_layer = Dense(10, activation='softmax')(dense_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model