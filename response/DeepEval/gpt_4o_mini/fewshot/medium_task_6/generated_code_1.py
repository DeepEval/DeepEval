import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Initial convolution layer
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Define the parallel blocks
    def parallel_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = keras.layers.Activation('relu')(batch_norm)
        return relu

    # Creating three parallel blocks
    block1_output = parallel_block(initial_conv)
    block2_output = parallel_block(initial_conv)
    block3_output = parallel_block(initial_conv)

    # Adding the outputs of the blocks to the initial convolution output
    added_output = Add()([initial_conv, block1_output, block2_output, block3_output])

    # Flattening the added output
    flatten_output = Flatten()(added_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_output)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model