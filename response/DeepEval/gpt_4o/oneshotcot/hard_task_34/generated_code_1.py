import keras
from keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def main_block(input_tensor):
        relu = ReLU()(input_tensor)
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(relu)
        concatenated = Concatenate()([input_tensor, sep_conv])
        return concatenated

    # Main path with three repeated blocks
    main_path = main_block(input_layer)
    main_path = main_block(main_path)
    main_path = main_block(main_path)

    # Branch path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), padding='same')(input_layer)

    # Fusion of main and branch paths
    fused = Add()([main_path, branch_path])

    # Flatten and output
    flatten_layer = Flatten()(fused)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model