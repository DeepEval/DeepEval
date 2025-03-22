import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Add, Flatten, Dense, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def split_and_extract(x):
        x1 = Conv2D(32, (1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x2 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x2 = BatchNormalization()(x2)
        x3 = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
        x3 = BatchNormalization()(x3)
        return tf.keras.layers.concatenate([x1, x2, x3], axis=-1)

    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    x = [split_and_extract(xi) for xi in x]
    x = Concatenate(axis=-1)(x)

    # Block 2
    path1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)

    path2 = AveragePooling2D((3, 3), strides=(1, 1))(x)
    path2 = Conv2D(64, (1, 1), padding='same', activation='relu')(path2)

    path3 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    path3_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(64, (3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate(axis=-1)([path3_1, path3_2])

    path4 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    path4 = Conv2D(64, (3, 3), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(64, (1, 3), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(64, (3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate(axis=-1)([path4_1, path4_2])

    output = Concatenate(axis=-1)([path1, path2, path3, path4])
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    model = Model(inputs=input_layer, outputs=output)
    return model

# Example usage:
# model = dl_model()
# model.summary()