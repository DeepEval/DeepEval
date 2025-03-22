import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_path_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x

    def branch_path_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return x

    # Main path
    x = main_path_block(input_layer)
    x = main_path_block(x)

    # Branch path
    branch = branch_path_block(input_layer)

    # Sum the outputs of the main and branch paths
    x = Add()([x, branch])

    # Flatten the result
    x = Flatten()(x)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model