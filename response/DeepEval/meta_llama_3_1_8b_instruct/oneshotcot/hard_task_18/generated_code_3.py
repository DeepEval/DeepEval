import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pooling = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)

    main_path = Add()([conv1, conv2, avg_pooling])
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    block_output = main_path

    def second_block(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(units=128, activation='relu')(path1)
        path1 = Dense(units=128, activation='relu')(path1)

        path2 = Multiply()([path1, input_tensor])
        path2 = Flatten()(path2)
        path2 = Dense(units=128, activation='relu')(path2)
        output_tensor = Dense(units=10, activation='softmax')(path2)

        return output_tensor

    output_layer = second_block(block_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model