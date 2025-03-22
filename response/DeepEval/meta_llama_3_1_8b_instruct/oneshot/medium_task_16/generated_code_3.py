import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))
    conv_split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    
    group1 = layers.Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_split[0])
    group2 = layers.Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_split[1])
    group3 = layers.Conv2D(filters=32 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_split[2])
    
    pool_split = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group1)
    pool_split = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group2)
    pool_split = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group3)
    
    concat_output = layers.Concatenate()([pool_split[0], pool_split[1], pool_split[2]])
    
    flatten_layer = layers.Flatten()(concat_output)
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model