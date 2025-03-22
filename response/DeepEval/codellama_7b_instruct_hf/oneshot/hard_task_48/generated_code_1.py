import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Concatenate()([x, x, x])
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=x)
    return model