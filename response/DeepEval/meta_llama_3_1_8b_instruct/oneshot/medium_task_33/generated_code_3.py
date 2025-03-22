import keras
from keras.layers import Input, Lambda, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    def channel_split(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=3)

    split_layer = Lambda(channel_split)(input_tensor=input_layer)
    
    conv1_1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv1_2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[0])
    conv1_3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[0])
    output1 = Concatenate()([conv1_1, conv1_2, conv1_3])
    
    conv2_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv2_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv2_3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[1])
    output2 = Concatenate()([conv2_1, conv2_2, conv2_3])
    
    conv3_1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[2])
    conv3_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[2])
    conv3_3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    output3 = Concatenate()([conv3_1, conv3_2, conv3_3])
    
    concatenated_output = Concatenate()([output1, output2, output3])
    bath_norm = BatchNormalization()(concatenated_output)
    
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model