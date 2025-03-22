import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, BatchNormalization, Reshape

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv1 = BatchNormalization()(conv1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2 = BatchNormalization()(conv2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3 = BatchNormalization()(conv3)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    def block_2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        path2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(path2)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        path3_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path3)
        path3_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path3)
        path3 = Concatenate()([path3_1, path3_2])
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(path4)
        path4_1 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path4)
        path4_2 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path4)
        path4 = Concatenate()([path4_1, path4_2])
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model