import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        path1 = Conv2D(kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(kernel_size=3, strides=1, padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(kernel_size=5, strides=1, padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    def block2(input_tensor):
        path1 = Conv2D(kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(kernel_size=1, strides=1, padding='same', activation='relu')(AveragePooling2D(pool_size=3, strides=1, padding='same')(input_tensor))
        path3 = Conv2D(kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        path3_1 = Conv2D(kernel_size=(1,3), strides=1, padding='same', activation='relu')(path3)
        path3_2 = Conv2D(kernel_size=(3,1), strides=1, padding='same', activation='relu')(path3)
        path4 = Conv2D(kernel_size=1, strides=1, padding='same', activation='relu')(input_tensor)
        path4_1 = Conv2D(kernel_size=(1,3), strides=1, padding='same', activation='relu')(path4)
        path4_2 = Conv2D(kernel_size=(3,1), strides=1, padding='same', activation='relu')(path4)
        output_tensor = Concatenate()([path1, path2, Concatenate()([path3_1, path3_2]), Concatenate()([path4_1, path4_2])])
        return output_tensor

    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    output_tensor1 = block1(split_input[0])
    output_tensor2 = block1(split_input[1])
    output_tensor3 = block1(split_input[2])
    concat_tensor = Concatenate()([output_tensor1, output_tensor2, output_tensor3])
    batch_norm = BatchNormalization()(concat_tensor)
    block_output = block2(batch_norm)
    flatten_layer = Flatten()(block_output)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model