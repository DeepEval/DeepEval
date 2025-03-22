# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # Define the same block
    def same_block(x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        weights = layers.Dense(128, activation='linear')(x)
        weights = layers.Reshape((1, 1, 128))(weights)
        return layers.multiply([x, weights])

    # Define the left branch
    left_branch = layers.Lambda(lambda x: x[:, :, :, :16])(inputs)
    left_branch = same_block(left_branch)

    # Define the right branch
    right_branch = layers.Lambda(lambda x: x[:, :, :, 16:])(inputs)
    right_branch = same_block(right_branch)

    # Concatenate the outputs from both branches
    concatenated = layers.Concatenate()([left_branch, right_branch])

    # Define the flattening layer
    flattened = layers.Flatten()(concatenated)

    # Define the output layer
    outputs = layers.Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model
model = dl_model()