import keras
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import preprocess_input

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) for _ in range(3)]
    x = [SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) for _ in range(3)]
    x = [SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x) for _ in range(3)]
    x = Concatenate()(x)

    # Branch path
    y = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    y = [Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(y) for _ in range(3)]
    y = Concatenate()(y)

    # Fuse outputs
    z = tf.add(x, y)

    # Flatten and FC layers
    z = Flatten()(z)
    z = Dense(units=128, activation='relu')(z)
    z = Dense(units=64, activation='relu')(z)
    output_layer = Dense(units=10, activation='softmax')(z)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model