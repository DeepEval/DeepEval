import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path block definition
    def main_path_block(input_tensor):
        # Using SeparableConv2D to extract features
        separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        relu_activation = Activation('relu')(separable_conv)
        return relu_activation

    # Creating the main path with three sequential blocks
    main_path_output = input_layer
    for _ in range(3):
        main_path_output = main_path_block(main_path_output)

    # Branch path
    branch_path_output = Conv2D(filters=main_path_output.shape[-1], kernel_size=(1, 1), padding='same')(input_layer)

    # Combine main path and branch path
    combined_output = Add()([main_path_output, branch_path_output])

    # Flatten and create fully connected layer
    flatten_output = Flatten()(combined_output)
    dense_output = Dense(units=128, activation='relu')(flatten_output)
    final_output = Dense(units=10, activation='softmax')(dense_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model