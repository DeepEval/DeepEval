import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial Convolutional Layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(input_layer)

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(initial_conv)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = ReLU()(block1_bn)

    # Block 2
    block2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(initial_conv)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = ReLU()(block2_bn)

    # Block 3
    block3_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='linear')(initial_conv)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = ReLU()(block3_bn)

    # Add the outputs of the three blocks with the initial convolution's output
    added_output = Add()([initial_conv, block1_relu, block2_relu, block3_relu])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model