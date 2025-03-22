import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, [16, 16, 16], axis=-1))(input_tensor)
        conv1 = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) for x in split1]
        conv1 = [BatchNormalization()形y] for x in conv1]
        pool1 = [MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x) for x in conv1]
        concat1 = Concatenate(axis=-1)(pool1)
        return concat1

    def block2(input_tensor):
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(conv2_1)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool2)
        conv2_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_3 = Lambda(lambda x: tf.split(x, [1, 3, 5], axis=-1))(conv2_3)
        conv2_3 = [tf.concat([conv2_3[0], conv2_3[1], conv2_3[2]], axis=-1) for conv2_3 in conv2_3]
        concat2 = Concatenate(axis=-1)([conv2_1] + conv2_3)
        return concat2

    input_layer = Input(shape=(32, 32, 3))
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model