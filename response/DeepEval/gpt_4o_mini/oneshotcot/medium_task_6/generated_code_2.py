import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # First block
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(initial_conv)
    block1_bn = BatchNormalization()(block1_conv)
    block1_out = ReLU()(block1_bn)

    # Second block
    block2_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(initial_conv)
    block2_bn = BatchNormalization()(block2_conv)
    block2_out = ReLU()(block2_bn)

    # Third block
    block3_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(initial_conv)
    block3_bn = BatchNormalization()(block3_conv)
    block3_out = ReLU()(block3_bn)

    # Add the outputs of the blocks to the initial convolution output
    added_output = Add()([initial_conv, block1_out, block2_out, block3_out])

    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model