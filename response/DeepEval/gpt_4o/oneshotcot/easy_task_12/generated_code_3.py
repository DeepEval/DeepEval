import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_path(input_tensor):
        # First Block
        x = Activation('relu')(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

        # Second Block
        x = Activation('relu')(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

        return x

    main_output = main_path(input_layer)

    # Branch Path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)

    # Summing Main and Branch Paths
    combined_output = Add()([main_output, branch_output])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model