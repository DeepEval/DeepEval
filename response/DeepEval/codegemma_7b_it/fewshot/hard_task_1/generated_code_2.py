import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Activation, Multiply, Add, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    def block_1(input_tensor):
        path1 = GlobalAveragePooling2D()(input_tensor)
        path1 = Dense(filters=4, activation='relu')(path1)
        path1 = Dense(filters=input_tensor.shape[3], activation='sigmoid')(path1)

        path2 = GlobalMaxPooling2D()(input_tensor)
        path2 = Dense(filters=4, activation='relu')(path2)
        path2 = Dense(filters=input_tensor.shape[3], activation='sigmoid')(path2)

        output_tensor = Add()([path1, path2])
        output_tensor = Multiply()([output_tensor, input_tensor])

        return output_tensor

    block1_output = block_1(conv1)

    # Block 2
    def block_2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        concat = keras.layers.concatenate([avg_pool, max_pool])
        conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
        conv = Activation('sigmoid')(conv)

        output_tensor = Multiply()([conv, input_tensor])

        return output_tensor

    block2_output = block_2(block1_output)

    # Final layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model