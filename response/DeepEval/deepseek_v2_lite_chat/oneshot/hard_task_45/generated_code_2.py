import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    def block1(input_tensor):

        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

        return maxpool1

    def block2(input_tensor):

        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
        conv5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv4)
        maxpool3 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv4)

        return Concatenate(axis=-1)([maxpool2, maxpool3, conv5])

    block1_output = block1(split[0])
    block2_output = block2(split[1])
    block2_output = block2(split[2])

    concat = Concatenate(axis=-1)([block1_output, block2_output])
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model