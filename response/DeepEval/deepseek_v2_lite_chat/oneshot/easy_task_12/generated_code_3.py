import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    x_test = x_test.reshape((len(x_test), 28, 28, 1))

    # Input layers
    input_layer_main = Input(shape=(28, 28, 1))
    input_layer_branch = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer_main)
    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer_main)
    conv2 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch path
    branch_conv = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer_branch)

    # Concatenate main and branch paths
    concat = Concatenate()([pool1, pool2, branch_conv])

    # Batch normalization and flattening
    batchnorm = BatchNormalization()(concat)
    flatten = Flatten()(batchnorm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=[input_layer_main, input_layer_branch], outputs=output_layer)

    return model