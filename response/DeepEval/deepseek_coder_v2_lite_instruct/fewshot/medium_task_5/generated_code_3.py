import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        return x

    main_path_output_1 = main_path(input_layer)
    main_path_output_2 = main_path(main_path_output_1)

    # Branch path
    def branch_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        return x

    branch_path_output = branch_path(input_layer)

    # Combining outputs
    combined_output = Add()([main_path_output_2, branch_path_output])

    # Flattening the output
    flattened_output = Flatten()(combined_output)

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model