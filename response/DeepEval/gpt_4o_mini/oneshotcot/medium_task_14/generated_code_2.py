import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32 with 3 color channels
    
    # Define the first block
    def conv_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # Create three blocks
    block1_output = conv_block(input_layer)
    block2_output = conv_block(block1_output)
    block3_output = conv_block(block2_output)

    # Direct parallel branch
    parallel_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_output = BatchNormalization()(parallel_output)
    parallel_output = Activation('relu')(parallel_output)

    # Sum the outputs of the three blocks and the parallel branch
    combined_output = Add()([block1_output, block2_output, block3_output, parallel_output])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model