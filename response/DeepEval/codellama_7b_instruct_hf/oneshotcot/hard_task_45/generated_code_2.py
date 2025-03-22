import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = Cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block of the model
    first_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    first_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(first_block)
    first_block = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(first_block)
    first_block = Flatten()(first_block)

    # Define the second block of the model
    second_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    second_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(second_block)
    second_block = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(second_block)
    second_block = Flatten()(second_block)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(second_block)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model