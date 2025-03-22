import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x[0])
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x[1])
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x[2])
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Second block
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model