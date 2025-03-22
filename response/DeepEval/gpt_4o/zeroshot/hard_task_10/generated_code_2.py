from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First feature extraction path: 1x1 convolution
    path1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Second feature extraction path: 1x1, 1x7, 7x1 convolutions
    path2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    path2 = Conv2D(32, (1, 7), activation='relu', padding='same')(path2)
    path2 = Conv2D(32, (7, 1), activation='relu', padding='same')(path2)

    # Concatenate the outputs of the two paths
    concatenated = Concatenate()([path1, path2])

    # Apply a 1x1 convolution to align the output dimensions
    main_path_output = Conv2D(3, (1, 1), activation='relu', padding='same')(concatenated)

    # Branch connecting directly to the input
    branch = input_layer

    # Merge the main path output and the branch through addition
    merged_output = Add()([main_path_output, branch])

    # Flatten the output for the fully connected layers
    flattened = Flatten()(merged_output)

    # Fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Usage example:
# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and compile the model
model = dl_model()

# Print model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))