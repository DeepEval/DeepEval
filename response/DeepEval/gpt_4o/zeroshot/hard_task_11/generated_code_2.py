import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset to get the input shape
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]
    num_classes = 10

    # Define the input
    inputs = Input(shape=input_shape)

    # Branch 1: Single 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)

    # Branch 2: 1x1 -> 1x3 -> 3x1 convolutions
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)
    branch2 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(branch2)

    # Concatenate outputs from branch 1 and branch 2
    concatenated = Concatenate()([branch1, branch2])

    # Followed by a 1x1 convolution
    main_pathway = Conv2D(filters=input_shape[2], kernel_size=(1, 1), activation='relu', padding='same')(concatenated)

    # Add input (skip connection) to main pathway
    added = Add()([inputs, main_pathway])

    # Flatten and add fully connected layers for classification
    flat = Flatten()(added)
    fc1 = Dense(units=128, activation='relu')(flat)
    fc2 = Dense(units=num_classes, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage:
model = dl_model()
model.summary()