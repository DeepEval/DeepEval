import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv_1x1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x))(split_input[0])
    conv_3x3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x))(split_input[1])
    conv_5x5 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x))(split_input[2])
    batch_norm_1 = Lambda(lambda x: BatchNormalization()(x))(Concatenate()([conv_1x1, conv_3x3, conv_5x5]))

    # Second block
    branch_1 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x))(input_layer)
    branch_2 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x))(branch_1)
    branch_3 = Lambda(lambda x: Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x))(branch_1)
    branch_4 = Lambda(lambda x: MaxPooling2D(pool_size=(2, 2), padding='same')(x))(input_layer)

    concat_output = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Classification layers
    flatten_layer = Flatten()(concat_output)
    dense_1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()