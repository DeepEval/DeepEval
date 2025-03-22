import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        flatten1 = Flatten()(path1)
        flatten2 = Flatten()(path2)
        flatten3 = Flatten()(path3)
        
        dropout = Dropout(0.2)
        concat = Concatenate()([dropout(flatten1), dropout(flatten2), dropout(flatten3)])
        
        flatten = Flatten()(concat)
        
        reshape = Reshape((512,))
        
        dense = Dense(units=128, activation='relu')(reshape)
        
        return dense
    
    block_output = block(input_layer)
    
    split = Lambda(lambda x: tf.split(x, 4, axis=-1))(block_output)
    
    path1 = layers.SeparableConv2D(32, (1, 1), activation='relu', padding='same')(split[0])
    path2 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(split[1])
    path3 = layers.SeparableConv2D(32, (5, 5), activation='relu', padding='same')(split[2])
    path4 = layers.SeparableConv2D(32, (7, 7), activation='relu', padding='same')(split[3])
    
    concat = Concatenate()([path1, path2, path3, path4])
    
    flatten = Flatten()(concat)
    
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model