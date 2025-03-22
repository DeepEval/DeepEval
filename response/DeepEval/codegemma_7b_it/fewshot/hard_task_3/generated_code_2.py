import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_main(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Dropout(0.2)(conv2)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        conv4 = Dropout(0.2)(conv4)
        conv5 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
        output_tensor = concatenate([conv2, conv4, conv6])
        return output_tensor

    def block_branch(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = conv1
        return output_tensor

    block1_output = block_main(input_tensor=input_layer)
    block2_output = block_branch(input_tensor=input_layer)
    combined_output = tf.keras.layers.add([block1_output, block2_output])

    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model