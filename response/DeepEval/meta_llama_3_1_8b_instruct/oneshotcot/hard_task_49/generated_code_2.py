import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    pooling1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pooling1_flat = Flatten()(pooling1)
    
    pooling2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pooling2_flat = Flatten()(pooling2)
    
    pooling3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pooling3_flat = Flatten()(pooling3)
    
    merged = Concatenate()([pooling1_flat, pooling2_flat, pooling3_flat])
    
    dense1 = Dense(units=128, activation='relu')(merged)
    reshape = Reshape((4, 128))(dense1)
    
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(reshape)
    
    group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
    group4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split[3])
    
    merged = Concatenate()([group1, group2, group3, group4])
    merged = GlobalAveragePooling2D()(merged)
    output_layer = Dense(units=10, activation='softmax')(merged)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model