import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def down_sample(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=tf.shape(input_tensor)[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=tf.shape(input_tensor)[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=tf.shape(input_tensor)[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        maxpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor

    block1_output = down_sample(input_tensor=input_layer)
    flatten = Flatten()(block1_output)
    dense1 = Dense(units=32, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model