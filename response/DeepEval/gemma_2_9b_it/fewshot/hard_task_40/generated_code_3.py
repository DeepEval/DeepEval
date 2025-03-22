import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Dropout, Reshape, Conv2D, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense1 = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense1)

    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        dropout1_1 = Dropout(rate=0.2)(conv1_1)

        conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        dropout2_1 = Dropout(rate=0.2)(conv2_2)
        
        conv3_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
        dropout3_1 = Dropout(rate=0.2)(conv3_2)

        avgpool = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[3])
        conv4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avgpool)
        dropout4_1 = Dropout(rate=0.2)(conv4_1)
        
        output_tensor = Concatenate(axis=-1)([dropout1_1, dropout2_1, dropout3_1, dropout4_1])

        return output_tensor

    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense2 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model