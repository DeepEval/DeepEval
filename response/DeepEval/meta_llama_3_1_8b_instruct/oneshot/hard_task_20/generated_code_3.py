import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, Add, BatchNormalization, Flatten, Dense
from keras.layers.core import Reshape
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split = Lambda(split_input)(input_layer)
    
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([group1, group2, group3])
    
    # Branch path
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the main path and the branch path
    fused_features = Add()([concatenated, branch])
    
    bath_norm = BatchNormalization()(fused_features)
    
    # Global average pooling to reduce spatial dimensions
    global_avg_pooling = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=True))(bath_norm)
    
    flatten_layer = Flatten()(global_avg_pooling)
    
    # Classification using two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model