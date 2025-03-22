import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images to [-1, 1]
    x_train = x_train / 127.5 - 1.
    x_test = x_test / 127.5 - 1.

    # Define the input shape
    input_shape = x_train[0].shape

    # Input layer
    inputs = Input(shape=input_shape)

    # First block
    def block1():
        # First convolutional layer
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
        # First batch normalization
        bn1 = BatchNormalization()(conv1)
        # First max pooling
        max_pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)
        # Second convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(max_pool1)
        # Second batch normalization
        bn2 = BatchNormalization()(conv2)
        # Second max pooling
        max_pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
        # Add the outputs of the two convolutional layers
        add1 = Add()([bn1, bn2])
        return add1

    # Second block
    def block2():
        # Global average pooling
        avg_pool = GlobalAveragePooling2D()(add1)
        # First fully connected layer
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        # Second fully connected layer
        dense2 = Dense(units=256, activation='relu')(dense1)
        # Output layer
        output = Dense(units=10, activation='softmax')(dense2)

        # Construct the model
        model = Model(inputs=inputs, outputs=output)
        return model

    # Create the model
    model = block1()
    model = block2()

    return model