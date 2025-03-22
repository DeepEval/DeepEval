import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    # First block
    block1_output = block(input_layer, filters=32, kernel_size=(3, 3))

    # Second block
    block2_output = block(block1_output, filters=64, kernel_size=(3, 3))

    # Third block
    block3_output = block(block2_output, filters=128, kernel_size=(3, 3))

    # Parallel path processing the input directly
    direct_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation=None)(input_layer)
    direct_output = BatchNormalization()(direct_output)
    direct_output = ReLU()(direct_output)

    # Adding the outputs from all paths
    added_output = Add()([block1_output, block2_output, block3_output, direct_output])

    # Flatten the result
    flatten_layer = Flatten()(added_output)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model