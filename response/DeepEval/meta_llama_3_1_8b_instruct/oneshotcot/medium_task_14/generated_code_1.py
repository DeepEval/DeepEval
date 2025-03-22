import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.layers import Add
from keras.models import Model
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    
    block1_output = batch_norm1
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    batch_norm2 = BatchNormalization()(conv2)
    
    block2_output = batch_norm2
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    batch_norm3 = BatchNormalization()(conv3)
    
    block3_output = batch_norm3
    
    parallel_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm_parallel = BatchNormalization()(parallel_branch)
    
    output_tensor = Concatenate()([block1_output, block2_output, block3_output, batch_norm_parallel])
    output_tensor = Add()([output_tensor, block3_output])
    
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model