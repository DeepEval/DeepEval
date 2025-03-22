import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = ReLU()(batch_norm)
        return relu

    # First block
    block1_output = block(input_tensor=input_layer, filters=32)

    # Second block, concatenate with output of first block
    concat1 = Concatenate()([input_layer, block1_output])
    block2_output = block(input_tensor=concat1, filters=64)

    # Third block, concatenate with output of second block
    concat2 = Concatenate()([concat1, block2_output])
    block3_output = block(input_tensor=concat2, filters=128)

    # Flatten and add fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model