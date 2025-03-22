import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_layer, filters=32)

    # Second block
    block2_output = block(block1_output, filters=64)

    # Third block
    block3_output = block(block2_output, filters=128)

    # Direct convolutional branch
    direct_branch = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    direct_branch = BatchNormalization()(direct_branch)
    direct_branch = ReLU()(direct_branch)

    # Add the outputs of all paths
    added_output = Add()([block1_output, block2_output, block3_output, direct_branch])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model