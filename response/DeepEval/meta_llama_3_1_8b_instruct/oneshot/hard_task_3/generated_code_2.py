import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Dropout, Dense
from keras.initializers import he_normal
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=-1)
    
    split_tensor = Lambda(split_input)(input_layer)
    
    pathway_outputs = []
    for i in range(3):
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[i])
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        dropout_layer = Dropout(0.2)(conv3x3)
        pathway_outputs.append(dropout_layer)
    
    concatenated_output = Concatenate()(pathway_outputs)
    
    branch_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = branch_conv1x1
    
    addition_output = Add()([concatenated_output, branch_output])
    
    bath_norm = BatchNormalization()(addition_output)
    flatten_layer = Flatten()(bath_norm)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model