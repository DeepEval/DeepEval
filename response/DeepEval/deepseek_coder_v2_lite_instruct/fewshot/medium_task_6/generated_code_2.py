import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolution
    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_initial = BatchNormalization()(conv_initial)
    relu_initial = ReLU()(batch_norm_initial)

    # Define the parallel blocks
    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # Apply each block to the initial convolution output
    block1_output = block(relu_initial)
    block2_output = block(relu_initial)
    block3_output = block(relu_initial)

    # Add the outputs of the blocks to the initial convolution output
    added_output = Add()([relu_initial, block1_output, block2_output, block3_output])

    # Flatten the result
    flattened_output = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)
    return model