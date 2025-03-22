import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_path_block(input_tensor):
        # Apply a separable convolution followed by ReLU activation
        sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        relu_activation = Activation('relu')(sep_conv)

        # Concatenate input with the output of the separable convolution
        concatenated_output = keras.layers.Concatenate()([input_tensor, relu_activation])
        
        return concatenated_output

    # Create the main path with three blocks
    main_path = input_layer
    for _ in range(3):
        main_path = main_path_block(main_path)

    # Branch Path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(1, 1), padding='same')(input_layer)

    # Fuse the two paths using addition
    fused_output = Add()([main_path, branch_path])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model