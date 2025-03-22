import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Lambda, DepthwiseConv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv_restore = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Branch Path
    branch_input = input_layer
    
    # Combine Main Path and Branch Path
    combine_output = Add()([conv_restore, branch_input])
    
    # Second Block
    def split_func(inputs):
        return tf.split(inputs, num_or_size_splits=3, axis=-1)
    
    split_layer = Lambda(split_func)(combine_output)
    
    group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    
    concat_output = Concatenate()([group1, group2, group3])
    
    # Process Features
    batch_norm = BatchNormalization()(concat_output)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm)
    
    flatten_layer = Flatten()(max_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model