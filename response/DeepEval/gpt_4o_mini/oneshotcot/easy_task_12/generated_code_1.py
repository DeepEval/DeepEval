import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # First block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Second block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        return x

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolution to match dimensions
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    # Get outputs from both paths
    main_output = main_path(input_layer)
    branch_output = branch_path(main_output)

    # Sum the outputs from both paths
    combined_output = Add()([main_output, branch_output])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model