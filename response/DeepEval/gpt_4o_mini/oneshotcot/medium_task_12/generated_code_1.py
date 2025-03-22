import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Concatenate

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    def conv_block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        norm = BatchNormalization()(conv)
        relu = ReLU()(norm)
        return relu

    # First block
    block1_output = conv_block(input_layer)

    # Second block
    block2_output = conv_block(block1_output)
    block2_output = Concatenate()([block1_output, block2_output])  # Concatenate with previous block

    # Third block
    block3_output = conv_block(block2_output)
    block3_output = Concatenate()([block2_output, block3_output])  # Concatenate with previous block

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model