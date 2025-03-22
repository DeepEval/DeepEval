import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Lambda, Reshape, Conv3D, MaxPooling3D
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        # Split the input into three groups using Lambda layer
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        output_tensor = Concatenate()([path1, path2, path3])

        return output_tensor

    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor))
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    block_output1 = block1(input_layer)
    batch_norm1 = BatchNormalization()(block_output1)
    block_output2 = block2(batch_norm1)
    batch_norm2 = BatchNormalization()(block_output2)
    flatten_layer = Flatten()(batch_norm2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model