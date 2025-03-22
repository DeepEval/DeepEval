import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path block
    def main_path_block(input_tensor):
        # Separable convolution layer with ReLU activation
        conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(conv)
        # Concatenate the input tensor with the output of the separable convolution
        output_tensor = Concatenate()([input_tensor, conv])
        return output_tensor

    # Process through the main path block three times
    main_path_output = input_layer
    for _ in range(3):
        main_path_output = main_path_block(main_path_output)

    # Branch path
    branch_path = Conv2D(filters=main_path_output.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Fuse the two paths
    fused_output = Add()([main_path_output, branch_path])

    # Flatten and classify
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model