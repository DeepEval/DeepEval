import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset image size is 32x32

    def conv_block(input_tensor, num_filters, kernel_size):
        return Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)

    main_path = input_layer
    main_path = conv_block(main_path, 32, (3, 3))
    main_path = conv_block(main_path, 64, (3, 3))
    main_path = conv_block(main_path, 128, (3, 3))

    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)

    adding_layer = Add()([main_path, branch_path])  # Add the main path and branch path

    path1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(adding_layer)

    path2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    adding_layer2 = Add()([path1, path2])

    flatten_layer = Flatten()(adding_layer2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model