import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ReLU, Concatenate, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import np_utils

def dl_model():
    # Number of classes
    num_classes = 10

    # Image dimensions
    img_rows, img_cols = 28, 28

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Define the main path
    input_main = Input(shape=input_shape)
    x = input_main

    # First block of the main path
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block of the main path
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    branch_input = Conv2D(64, (1, 1), activation='relu')(x)

    # Concatenate outputs from main and branch paths
    concat = Concatenate()([x, branch_input])

    # Flatten and pass through a fully connected layer
    x = Flatten()(concat)
    output = Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_main, branch_input], outputs=[output])

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Instantiate and return the model
model = dl_model()