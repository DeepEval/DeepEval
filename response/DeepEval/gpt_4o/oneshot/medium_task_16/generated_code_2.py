import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group
    conv_group1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv_group2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv_group3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    
    # Downsample each group via average pooling
    pool_group1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group1)
    pool_group2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group2)
    pool_group3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group3)
    
    # Concatenate the feature maps
    concatenated = Concatenate(axis=-1)([pool_group1, pool_group2, pool_group3])
    
    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model