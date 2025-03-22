import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    # Input layer with 1 channel
    input_layer = Input(shape=(28, 28, 1))

    # Convolutional layer with 32 filters and a kernel size of 3x3
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Max pooling layer with a pool size of 2x2 and strides of 2x2
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Convolutional layer with 64 filters and a kernel size of 3x3
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)

    # Max pooling layer with a pool size of 2x2 and strides of 2x2
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Add the outputs of both paths
    adding_layer = Add()([pool1, pool2])

    # Flatten the output
    flatten_layer = Flatten()(adding_layer)

    # Dense layer with 128 units and a softmax activation function
    dense_layer = Dense(units=128, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model