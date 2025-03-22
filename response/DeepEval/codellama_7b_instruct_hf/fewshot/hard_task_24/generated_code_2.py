import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

 和 return model
def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    model = keras.Sequential([
        # Initial convolutional layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # First branch
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        UpSampling2D((2, 2)),
        # Second branch
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        UpSampling2D((2, 2)),
        # Third branch
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        UpSampling2D((2, 2)),
        # Fuse the outputs of all branches
        Concatenate()([model.output, model.output, model.output]),
        # Final convolutional layer
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model