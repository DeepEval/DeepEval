from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the model
    model = Sequential()

    # Add the first convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    # Add the first average pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Add the second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Add the second average pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Add the third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Add the third average pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Add the fourth convolutional layer
    model.add(Conv2D(256, (3, 3), activation='relu'))

    # Add the fourth average pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Add the flattening layer
    model.add(Flatten())

    # Add the first fully connected layer
    model.add(Dense(128, activation='relu'))

    # Add the dropout layer
    model.add(Dropout(0.5))

    # Add the second fully connected layer
    model.add(Dense(128, activation='relu'))

    # Add the dropout layer
    model.add(Dropout(0.5))

    # Add the output layer
    model.add(Dense(1000, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model