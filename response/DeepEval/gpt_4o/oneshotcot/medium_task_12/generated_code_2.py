import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', activation=None)(input_tensor)
        bn = BatchNormalization()(conv)
        relu = ReLU()(bn)
        return relu
    
    # First block
    block1_output = block(input_layer, filters=32)
    
    # Second block, concatenating with the output of the first block
    block2_input = Concatenate()([input_layer, block1_output])
    block2_output = block(block2_input, filters=64)
    
    # Third block, concatenating with the output of the second block
    block3_input = Concatenate()([block2_input, block2_output])
    block3_output = block(block3_input, filters=128)
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(block3_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model