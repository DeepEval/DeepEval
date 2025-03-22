import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        unchanged_group = inputs_groups[0]
        second_group = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        combined_group = Concatenate()([unchanged_group, second_group])
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
        output_tensor = Concatenate()([unchanged_group, second_group, conv])
        return output_tensor

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    main_output = main_path(input_layer)
    branch_output = branch_path

    adding_layer = Add()([main_output, branch_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model