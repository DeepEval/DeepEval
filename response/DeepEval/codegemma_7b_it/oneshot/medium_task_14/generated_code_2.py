import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_act = Activation('relu')(block1_bn)
    block1_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_act)
    
    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1_max_pool)
    block2_bn = BatchNormalization()(block2_conv)
    block2_act = Activation('relu')(block2_bn)
    block2_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2_act)
    
    # Block 3
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2_max_pool)
    block3_bn = BatchNormalization()(block3_conv)
    block3_act = Activation('relu')(block3_bn)
    
    # Parallel branch
    parallel_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    parallel_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    parallel_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_layer)
    parallel_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    
    # Concatenate outputs
    concat = concatenate([block3_act, parallel_conv1, parallel_conv2, parallel_conv3, parallel_max_pool])
    
    # Fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model