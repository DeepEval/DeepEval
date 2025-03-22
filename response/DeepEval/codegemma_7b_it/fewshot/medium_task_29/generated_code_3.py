import keras
import tensorflow as tf
from keras.layers import Input, MaxPooling2D, Lambda, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = tf.concat([flatten1, flatten2, flatten3], axis=-1)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 16))(dense)
    
    # Second block
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(reshaped)
    conv1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[0])
    conv2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs_groups[1])
    conv3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(inputs_groups[2])
    conv4 = MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(inputs_groups[3])
    output_tensor = tf.concat([conv1, conv2, conv3, conv4], axis=-1)

    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model