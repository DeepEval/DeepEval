import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block_main(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool1

    main_path = block_main(input_layer)
    main_path = block_main(main_path)

    # Branch path
    def block_branch(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        return pool1

    branch_path = block_branch(input_layer)

    # Addition operation
    adding_layer = Add()([main_path, branch_path])

    # Flatten layer
    flatten_layer = Flatten()(adding_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model