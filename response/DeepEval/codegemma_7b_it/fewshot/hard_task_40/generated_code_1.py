import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Dropout

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor = Concatenate()([maxpool1, maxpool2, maxpool3])
        return output_tensor

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        path1 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(inputs_groups[0])
        path1 = Dropout(0.1)(path1)
        path2 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(inputs_groups[1])
        path2 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(path2)
        path2 = Dropout(0.1)(path2)
        path3 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x))(inputs_groups[2])
        path3 = Dropout(0.1)(path3)
        path4 = Lambda(lambda x: DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x))(inputs_groups[3])
        path4 = Lambda(lambda x: AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x))(path4)
        path4 = Dropout(0.1)(path4)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense)
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model