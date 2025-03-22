import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First branch
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Second branch
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Third branch
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(branch3)

    # Fourth branch
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1)(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate outputs of the branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated features
    flattened = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for CIFAR-10

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model