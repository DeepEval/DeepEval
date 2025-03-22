import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, Add, BatchNormalization, Flatten, Dense
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
import tensorflow as tf
from keras import regularizers

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        group1, group2, group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        concat_output = Concatenate()([group1, group2, group3])
        output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat_output)
        return output

    def branch_path(input_tensor):
        output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output

    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    fused_output = Add()([main_output, branch_output])
    bath_norm = BatchNormalization()(fused_output)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model