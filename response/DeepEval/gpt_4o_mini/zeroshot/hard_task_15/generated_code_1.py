import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten, Add, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x_main = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x_main = Conv2D(64, (3, 3), padding='same', activation='relu')(x_main)
    x_main = GlobalAveragePooling2D()(x_main)

    # Fully connected layers in main path
    x_main = Dense(128, activation='relu')(x_main)
    x_main = Dense(32, activation='relu')(x_main)

    # Reshape weights to match input layer's shape
    x_main = Reshape((1, 1, 32))(x_main)
    x_main = Conv2D(3, (1, 1), padding='same')(x_main)  # Reshape to match the input channels
    x_main = Add()([x_main, input_layer])  # Element-wise addition with the input layer

    # Branch path (direct connection to the input layer)
    x_branch = input_layer

    # Combine main and branch paths
    combined = Add()([x_main, x_branch])

    # Fully connected layers after combining paths
    x_combined = GlobalAveragePooling2D()(combined)
    x_combined = Dense(128, activation='relu')(x_combined)
    x_combined = Dense(10, activation='softmax')(x_combined)  # Final output layer for 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=x_combined)

    return model

# Example of how to instantiate the model
model = dl_model()
model.summary()