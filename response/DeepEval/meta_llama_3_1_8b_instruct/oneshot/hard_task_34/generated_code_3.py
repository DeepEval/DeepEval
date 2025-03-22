import keras
from keras.layers import Input, SeparableConv2D, ReLU, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor, num_blocks=3):
        output_tensor = input_tensor
        for _ in range(num_blocks):
            conv = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(output_tensor)
            concat = Concatenate()([output_tensor, conv])
            output_tensor = BatchNormalization()(concat)
        return output_tensor
    
    main_path = block(input_tensor=input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    fusion = Add()([main_path, branch_path])
    bath_norm = BatchNormalization()(fusion)
    block_2 = block(input_tensor=bath_norm)
    block_3 = block(input_tensor=block_2)
    block_4 = block(input_tensor=block_3)
    
    flatten_layer = Flatten()(block_4)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model