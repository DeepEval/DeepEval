import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense, DepthwiseConv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat_layer = Concatenate()([input_tensor, conv1])
        output_tensor = conv1
        return output_tensor
    
    def branch_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path = main_path_block(input_layer)
    for _ in range(3):
        main_path = main_path_block(main_path)

    branch_path_output = branch_path(input_layer)
    adding_layer = Add()([main_path, branch_path_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model