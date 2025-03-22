import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    model = keras.Sequential()

    # Add the input layer
    model.add(Input(shape=input_shape))

    # Add the convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    # Add the max pooling layer
    model.add(MaxPooling2D((2, 2)))

    # Define the parallel paths
    parallel_paths = []

    # First path: 1x1 convolution
    parallel_paths.append(Conv2D(32, (1, 1), activation='relu'))

    # Second path: 3x3 convolutions
    parallel_paths.append(Conv2D(32, (3, 3), activation='relu'))
    parallel_paths.append(Conv2D(32, (3, 3), activation='relu'))

    # Third path: 3x3 convolution
    parallel_paths.append(Conv2D(32, (3, 3), activation='relu'))

    # Fourth path: max pooling
    parallel_paths.append(MaxPooling2D((2, 2)))

    # Concatenate the outputs from the parallel paths
    model.add(Concatenate(parallel_paths))

    # Add batch normalization
    model.add(BatchNormalization())

    # Add flatten layer
    model.add(Flatten())

    # Add dense layer
    model.add(Dense(128, activation='relu'))

    # Add output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model