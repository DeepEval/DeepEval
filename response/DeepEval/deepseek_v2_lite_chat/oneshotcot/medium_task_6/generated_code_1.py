import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ReLU

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # Cifr-10 image size is 32x32

    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define a block
    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        relu = ReLU()(batch_norm)
        return relu

    # First block
    block1_output = block(conv1)
    
    # Second block
    block2_output = block(block1_output)
    
    # Third block
    block3_output = block(block2_output)

    # Concatenate outputs of the blocks
    concat_output = Concatenate()([block1_output, block2_output, block3_output, conv1])

    # Flatten and pass through dense layers
    flatten = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model