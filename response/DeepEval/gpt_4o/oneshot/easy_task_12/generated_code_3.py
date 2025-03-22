import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense, ReLU

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_path(input_tensor):
        # Block 1
        x = ReLU()(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        # Block 2
        x = ReLU()(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        
        return x

    # Branch path
    def branch_path(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_tensor)
        return x

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)

    # Sum the outputs of the main and branch paths
    summed_output = Add()([main_output, branch_output])

    # Flatten and fully connected layer for final classification
    flatten_layer = Flatten()(summed_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model