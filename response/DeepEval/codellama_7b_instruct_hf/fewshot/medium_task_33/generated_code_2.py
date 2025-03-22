import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def split_channels(x):
        return tf.split(x, 3, axis=-1)

    def extract_features(x):
        x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    x1 = Lambda(split_channels)(input_layer)
    x2 = Lambda(extract_features)(x1[0])
    x3 = Lambda(extract_features)(x1[1])
    x4 = Lambda(extract_features)(x1[2])

    x = Concatenate()([x2, x3, x4])
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model