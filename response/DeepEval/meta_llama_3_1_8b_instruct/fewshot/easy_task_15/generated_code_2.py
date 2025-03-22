import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        drop = Dropout(rate=0.2)(pool)
        return drop

    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=block1_output)

    pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='same')(block2_output)

    flatten = Flatten()(pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model