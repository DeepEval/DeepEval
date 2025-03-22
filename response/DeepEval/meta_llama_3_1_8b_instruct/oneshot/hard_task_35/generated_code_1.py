import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, BatchNormalization, Flatten
from keras.layers import add, Lambda
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def branch(input_tensor):

        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        avg_pool = GlobalAveragePooling2D()(conv1)
        dense1 = Dense(units=32, activation='relu')(avg_pool)
        weights = Dense(units=3, activation='linear')(dense1)
        weights = Reshape(target_shape=(1, 1, 3))(weights)
        
        return Multiply()([conv1, weights])
        
    branch1_output = branch(input_layer)
    branch2_output = branch(branch1_output)
    
    bath_norm1 = BatchNormalization()(branch1_output)
    bath_norm2 = BatchNormalization()(branch2_output)
    concat_layer = Concatenate()([bath_norm1, bath_norm2])
    flatten_layer = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model