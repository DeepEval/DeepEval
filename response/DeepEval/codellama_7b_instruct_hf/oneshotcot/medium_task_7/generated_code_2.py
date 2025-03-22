from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.cifar10 import Cifar10


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = Cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(x_train)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(conv2)

    # Add the outputs of the first two convolutional layers
    added_conv1_conv2 = Concatenate()([conv1, conv2])

    # Add the output of the third convolutional layer
    added_conv1_conv2_conv3 = Concatenate()([added_conv1_conv2, conv3])

    # Define the fully connected layers
    fc1 = Dense(64, activation='relu')(added_conv1_conv2_conv3)
    fc2 = Dense(128, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=conv1.input, outputs=fc3)

    return model