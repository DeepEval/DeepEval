import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def pool_layer(input_tensor, window_size, stride):
        return MaxPooling2D(pool_size=(window_size, window_size), strides=(stride, stride), padding='same')(input_tensor)

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = pool_layer(input_tensor=conv1, window_size=1, stride=1)
    pool1_flatten = Flatten()(pool1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool2 = pool_layer(input_tensor=conv2, window_size=2, stride=2)
    pool2_flatten = Flatten()(pool2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool3 = pool_layer(input_tensor=conv3, window_size=4, stride=4)
    pool3_flatten = Flatten()(pool3)

    concat = Concatenate()([pool1_flatten, pool2_flatten, pool3_flatten])

    dense1 = Dense(units=512, activation='relu')(concat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model