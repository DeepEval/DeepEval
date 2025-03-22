import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def main_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        return x

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    main_output = main_path(input_layer)
    addition_output = Add()([main_output, branch_path])

    # Second block
    def separable_block(x, kernel_size):
        x = SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        return x

    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(addition_output)
    output_1x1 = separable_block(split_layer, kernel_size=(1, 1))
    output_3x3 = separable_block(split_layer, kernel_size=(3, 3))
    output_5x5 = separable_block(split_layer, kernel_size=(5, 5))

    concat_output = tf.concat([output_1x1, output_3x3, output_5x5], axis=-1)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()