import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # Block 1
        conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

        # Block 2
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

        return pool2

    # Branch path
    def branch_path(input_tensor):
        conv_branch = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv_branch

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)

    # Match dimensions of both paths
    branch_path_output = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(branch_path_output)

    # Add outputs from both paths
    combined_output = Add()([main_path_output, branch_path_output])

    # Flattening and Fully Connected Layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model