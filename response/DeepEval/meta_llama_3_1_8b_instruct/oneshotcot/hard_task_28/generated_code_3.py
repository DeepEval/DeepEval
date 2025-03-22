import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Add, LayerNormalization, Flatten, Dense
from keras.regularizers import l2
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same')(input_layer)
    main_path = LayerNormalization()(main_path)
    main_path = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine the two paths
    combined_path = Add()([main_path, branch_path])
    
    # Flatten the combined path
    flatten_layer = Flatten()(combined_path)
    
    # Classification layers
    dense1 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(flatten_layer)
    dense2 = Dense(units=10, activation='softmax', kernel_regularizer=l2(0.01))(dense1)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model