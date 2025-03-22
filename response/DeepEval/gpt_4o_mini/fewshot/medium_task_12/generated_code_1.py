import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are of size 32x32 with 3 color channels

    def conv_block(input_tensor, filters, kernel_size):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # Block 1
    block1_output = conv_block(input_layer, filters=32, kernel_size=(3, 3))

    # Block 2
    block2_output = conv_block(block1_output, filters=64, kernel_size=(3, 3))
    block2_output = Concatenate()([block1_output, block2_output])  # Concatenate with Block 1 output

    # Block 3
    block3_output = conv_block(block2_output, filters=128, kernel_size=(3, 3))
    block3_output = Concatenate()([block2_output, block3_output])  # Concatenate with Block 2 output

    # Flatten and Fully Connected Layers
    flatten = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model