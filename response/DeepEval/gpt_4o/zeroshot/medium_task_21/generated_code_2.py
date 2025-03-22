import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 1x1 Convolution
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: 1x1 Convolution followed by two consecutive 3x3 Convolutions
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.3)(branch3)

    # Branch 4: Average Pooling followed by 1x1 Convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(32, (1, 1), activation='relu')(branch4)
    branch4 = Dropout(0.3)(branch4)

    # Concatenate all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated feature map
    flatten_layer = Flatten()(concatenated_branches)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flatten_layer)
    fc2 = Dense(128, activation='relu')(fc1)
    fc3 = Dense(64, activation='relu')(fc2)

    # Output layer
    output_layer = Dense(10, activation='softmax')(fc3)  # CIFAR-10 has 10 classes

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()