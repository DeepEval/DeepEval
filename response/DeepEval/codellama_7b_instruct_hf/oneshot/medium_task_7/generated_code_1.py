import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the inputs
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)(X_train)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape)(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape)(conv2)

    # Add the outputs of the first two convolutional layers
    added_outputs = Concatenate()([conv1, conv2])

    # Add the output of the third convolutional layer
    added_outputs = Concatenate()([added_outputs, conv3])

    # Apply batch normalization
    batch_norm = BatchNormalization()(added_outputs)

    # Flatten the output
    flattened_output = Flatten()(batch_norm)

    # Define the first fully connected layer
    fc1 = Dense(128, activation='relu')(flattened_output)

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=X_train, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model