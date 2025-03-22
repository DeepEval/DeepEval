import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv_init = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Block 1
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_init)
    batch_norm1 = BatchNormalization()(conv_block1)
    relu_block1 = ReLU()(batch_norm1)

    # Block 2
    conv_block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_block1)
    batch_norm2 = BatchNormalization()(conv_block2)
    relu_block2 = ReLU()(batch_norm2)

    # Block 3
    conv_block3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu_block2)
    batch_norm3 = BatchNormalization()(conv_block3)
    relu_block3 = ReLU()(batch_norm3)

    # Add outputs of blocks to initial convolution
    concat_blocks = Concatenate()([conv_init, relu_block1, relu_block2, relu_block3])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concat_blocks)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model