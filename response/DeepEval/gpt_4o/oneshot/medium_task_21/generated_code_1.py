import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    branch3 = Dropout(0.3)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch4)
    branch4 = Dropout(0.3)(branch4)

    # Concatenating outputs of all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concatenated_branches)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dense1)
    dense2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model