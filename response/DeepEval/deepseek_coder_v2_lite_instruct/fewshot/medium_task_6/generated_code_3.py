import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolution layer
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # Three parallel blocks
    block1_output = block(conv_initial)
    block2_output = block(conv_initial)
    block3_output = block(conv_initial)

    # Add the outputs of the blocks to the initial convolution's output
    added_output = Add()([conv_initial, block1_output, block2_output, block3_output])

    # Flatten the result
    flattened_output = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model