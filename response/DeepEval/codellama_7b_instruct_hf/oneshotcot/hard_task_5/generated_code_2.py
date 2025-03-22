import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Concatenate()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=x)
    return model