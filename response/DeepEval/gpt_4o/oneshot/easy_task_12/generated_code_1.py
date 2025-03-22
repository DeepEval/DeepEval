import keras
from keras.layers import Input, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense, Activation

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_block(input_tensor):
        x = Activation('relu')(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x

    main_path = main_block(input_layer)
    main_path = main_block(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)

    # Sum the outputs of the main and branch paths
    summed = Add()([main_path, branch_path])

    # Flatten and Fully Connected Layer
    flatten_layer = Flatten()(summed)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()