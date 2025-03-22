import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = inputs_groups[0]
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tf.add(inputs_groups[1], inputs_groups[2]))
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor

    def branch_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    adding_layer = Add()([main_path_output, branch_path_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model