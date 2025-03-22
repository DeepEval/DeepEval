import keras
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Reshape, Activation, concatenate, add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Extract spatial features with a 7x7 depthwise separable convolutional layer
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', use_bias=False, depthwise_initializer='he_normal')(input_layer)
    
    # Layer normalization to enhance training stability
    conv_bn = BatchNormalization()(conv)
    
    # Activation function
    conv_act = Activation('relu')(conv_bn)
    
    # Two fully connected layers for channel-wise feature transformation
    dense1 = Dense(filters=64, kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(conv_act)
    dense1_bn = BatchNormalization()(dense1)
    dense1_act = Activation('relu')(dense1_bn)
    
    dense2 = Dense(filters=64, kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(dense1_act)
    dense2_bn = BatchNormalization()(dense2)
    dense2_act = Activation('relu')(dense2_bn)
    
    # Combine original input with processed features through an addition operation
    concat = concatenate([dense2_act, conv_act])
    
    # Final fully connected layers for classification
    dense3 = Dense(filters=64, kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(concat)
    dense3_bn = BatchNormalization()(dense3)
    dense3_act = Activation('relu')(dense3_bn)
    
    dense4 = Dense(filters=10, kernel_size=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(dense3_act)
    dense4_bn = BatchNormalization()(dense4)
    dense4_act = Activation('softmax')(dense4_bn)
    
    # Model output
    output_layer = dense4_act
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model