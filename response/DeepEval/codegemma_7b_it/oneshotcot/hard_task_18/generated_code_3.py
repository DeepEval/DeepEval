import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Dropout, concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        return avg_pool

    def block2(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        avg_pool = GlobalAveragePooling2D()(conv4)
        avg_pool = Dense(units=32, activation='relu')(avg_pool)
        avg_pool = Dense(units=64, activation='relu')(avg_pool)
        reshape = keras.layers.Reshape((1, 1, 64))(avg_pool)
        upsample = keras.layers.UpSampling2D((4, 4))(reshape)
        concat = concatenate([upsample, input_tensor])
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
        conv6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
        return conv6

    block1_output = block1(input_layer)
    block2_output = block2(block1_output)

    flatten = Flatten()(block2_output)
    dense = Dense(units=64, activation='relu')(flatten)
    dense = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model