import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Add, Flatten, Dense, Reshape, Lambda, Concatenate, Dropout
from keras.regularizers import l2
from keras.initializers import HeUniform


def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv_1x1_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(conv_1x1_main)
    
    # Branch path
    branch_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_1x1_main)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(branch_2)
    branch_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(branch_2)
    branch_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(branch_2)
    branch_2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(branch_2)

    # Concatenate outputs of both branches
    concat_layer = Concatenate()([branch_1, branch_2])
    
    # Main path output
    conv_1x1_main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(concat_layer)
    
    # Branch path output
    conv_1x1_branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=HeUniform(), kernel_regularizer=l2(0.0001))(concat_layer)
    
    # Fuse outputs and final classification
    merged_outputs = Add()([conv_1x1_main_output, conv_1x1_branch_output])
    flatten_layer = Flatten()(merged_outputs)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model