import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the model
    model = keras.models.Model(inputs=Input(shape=input_shape), outputs=None)

    # Add the first convolutional layer
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the second convolutional layer
    model.add(Conv2D(64, kernel_size=(3, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the third convolutional layer
    model.add(Conv2D(64, kernel_size=(1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add the dropout layers
    model.add(Dropout(0.2))
    model.add(Dropout(0.2))

    # Add the addition layer
    model.add(Add())

    # Add the flattening layer
    model.add(Flatten())

    # Add the fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model