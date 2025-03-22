import keras
from keras.layers import Input, Conv2D, SeparableConv2D, ReLU, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_block(input_tensor):
        relu = ReLU()(input_tensor)
        sep_conv = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(relu)
        output_tensor = Concatenate(axis=-1)([input_tensor, sep_conv])
        return output_tensor

    # Main path
    main_output = input_layer
    for _ in range(3):
        main_output = main_block(main_output)

    # Branch path
    branch_conv = Conv2D(filters=main_output.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse paths
    fused_output = Add()([main_output, branch_conv])

    # Final classification layers
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model