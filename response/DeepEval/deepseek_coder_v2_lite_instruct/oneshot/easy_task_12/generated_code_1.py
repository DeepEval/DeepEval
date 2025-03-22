import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    def main_block(input_tensor):
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x

    main_output = main_block(input_layer)

    # Branch path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Sum the outputs of the main and branch paths
    combined_output = Concatenate()([main_output, branch_output])

    # Flatten the result
    flatten_layer = Flatten()(combined_output)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model