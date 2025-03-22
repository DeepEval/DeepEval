import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Define the three parallel blocks
    def parallel_block(input_tensor):
        conv_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm_block = BatchNormalization()(conv_block)
        relu_block = ReLU()(batch_norm_block)
        return relu_block
    
    block1_output = parallel_block(initial_conv)
    block2_output = parallel_block(initial_conv)
    block3_output = parallel_block(initial_conv)

    # Add the outputs of the blocks to the initial convolution's output
    added_output = Add()([initial_conv, block1_output, block2_output, block3_output])

    # Flatten the added output and pass through fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model