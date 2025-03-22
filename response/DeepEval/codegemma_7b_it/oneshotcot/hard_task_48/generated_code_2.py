import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        group_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_tensor)
        conv_1 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x))(group_1[0])
        conv_2 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x))(group_1[1])
        conv_3 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x))(group_1[2])
        bn_1 = Lambda(lambda x: BatchNormalization()(x))(conv_1)
        bn_2 = Lambda(lambda x: BatchNormalization()(x))(conv_2)
        bn_3 = Lambda(lambda x: BatchNormalization()(x))(conv_3)
        concat_1 = Lambda(lambda x: Concatenate()([x[0], x[1], x[2]]))([bn_1, bn_2, bn_3])
        return concat_1

    block_1_output = block_1(input_tensor)

    # Block 2
    def block_2(input_tensor):
        path_1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_tensor)
        path_2 = Lambda(lambda x: MaxPooling2D(pool_size=(3, 3), padding='same')(x))(input_tensor)
        path_2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(path_2))(path_2)
        path_3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_tensor)
        path_3_1 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 3), padding='same', activation='relu')(x))(path_3)
        path_3_2 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(3, 1), padding='same', activation='relu')(x))(path_3)
        concat_2 = Lambda(lambda x: Concatenate()([x[0], x[1], x[2]])([path_1, path_2, path_3_1, path_3_2]))
        return concat_2

    block_2_output = block_2(block_1_output)

    # Output layer
    flatten_layer = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model