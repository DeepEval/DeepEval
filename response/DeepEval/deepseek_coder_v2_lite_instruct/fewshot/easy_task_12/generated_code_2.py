import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main Path
    def main_path(input_tensor):
        # First block
        conv1 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        # Second block
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool2

    main_path_output = main_path(input_layer)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Merge paths
    added_output = Add()([main_path_output, branch_path])
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model