import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path_block(input_tensor):
        # Separable convolution layer
        separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        # Concatenate input with the separable convolution output
        concatenated = Concatenate()([input_tensor, separable_conv])
        return concatenated

    # Repeat the main path block three times
    main_path_output = input_layer
    for _ in range(3):
        main_path_output = main_path_block(main_path_output)

    # Branch path
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Merge the two paths using addition
    merged_output = Add()([main_path_output, branch_path_output])

    # Flatten the merged output and pass it to a dense layer
    flatten_layer = Flatten()(merged_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model