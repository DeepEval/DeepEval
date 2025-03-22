import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, Add, Flatten, Dense, ReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Apply ReLU activation
        relu = ReLU()(input_tensor)
        # Apply Separable Convolution
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu)
        # Concatenate input and convolved feature map
        output_tensor = Concatenate()([input_tensor, sep_conv])

        return output_tensor

    # Main path
    main_path = block(input_layer)
    main_path = block(main_path)
    main_path = block(main_path)

    # Branch path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Fuse paths with addition
    fused_path = Add()([main_path, branch_path])

    # Flatten and output layer
    flatten_layer = Flatten()(fused_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model