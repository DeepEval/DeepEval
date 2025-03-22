from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Convert the labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)

    # Define the concatenate layer
    concatenate = Concatenate(axis=3)([conv1, conv2, conv3])

    # Define the flatten layer
    flatten = Flatten()(concatenate)

    # Define the first fully connected layer
    fc1 = Dense(128, activation='relu')(flatten)

    # Define the second fully connected layer
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))

    return model