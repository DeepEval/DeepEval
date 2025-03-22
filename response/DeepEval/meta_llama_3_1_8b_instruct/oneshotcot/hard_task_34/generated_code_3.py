import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    block_output = None

    def block(input_tensor):

        conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([input_tensor, conv])

        return output_tensor
    
    block_output = block(input_layer)
    block_output = block(block_output)
    block_output = block(block_output)

    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block_output)
    fusion_path = Add()([block_output, branch_path])
    bath_norm = BatchNormalization()(fusion_path)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model