import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 input shape
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # First path: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    # Second path: 1x1, 1x7, 7x1 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path2)

    # Concatenate the two paths
    concatenated = Concatenate()([path1, path2])

    # Apply a 1x1 convolution to the concatenated output
    main_path = Conv2D(3, (1, 1), activation='relu', padding='same')(concatenated)

    # Add a direct connection from the input
    merged = Add()([main_path, inputs])

    # Flatten the output
    flat = Flatten()(merged)

    # Fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flat)
    outputs = Dense(num_classes, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Load CIFAR-10 data (to be used when training the model)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Create the model
model = dl_model()

# Print the model summary
model.summary()