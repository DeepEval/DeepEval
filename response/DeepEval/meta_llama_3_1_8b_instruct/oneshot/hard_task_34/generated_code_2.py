import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    block_output = main_path(input_layer)

    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)
    fusion = Add()([block_output, branch_path])
    bath_norm = BatchNormalization()(fusion)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def main_path(input_tensor, filters=128):

    def block(input_tensor):
        sep_conv = SeparableConv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(sep_conv)
        output_tensor = Concatenate()([input_tensor, conv])

        return output_tensor
        
    block_output = block(input_tensor)
    block_output = block(block_output)
    block_output = block(block_output)

    return block_output