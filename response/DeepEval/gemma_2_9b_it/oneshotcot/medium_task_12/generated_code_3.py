import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_output = block1_bn

    # Block 2
    block2_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block2_bn = BatchNormalization()(block2_conv)
    block2_output = block2_bn

    # Block 3
    block3_conv = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block3_bn = BatchNormalization()(block3_conv)
    block3_output = block3_bn

    # Concatenate outputs
    concatenated_output = Concatenate(axis=3)([block1_output, block2_output, block3_output])

    # Flatten and dense layers
    flatten_layer = Flatten()(concatenated_output)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model