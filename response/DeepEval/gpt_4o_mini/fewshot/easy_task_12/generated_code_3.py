import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer for the MNIST dataset
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # Block 1
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

        # Block 2
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

        return x

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolution to match output dimensions
        return SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

    # Process input through main path
    main_output = main_path(input_layer)

    # Process input through branch path
    branch_output = branch_path(main_output)

    # Summing the outputs from main and branch paths
    merged_output = Add()([main_output, branch_output])

    # Flattening layer
    flatten_layer = Flatten()(merged_output)

    # Fully connected layer for output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model