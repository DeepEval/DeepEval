import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input along the channel dimension into three groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    group_convs = []
    num_kernels = input_layer.shape[-1] // 3  # Number of kernels for each group
    
    # Apply 1x1 convolution to each group independently
    for group in split_channels:
        conv = Conv2D(filters=num_kernels, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        pooled = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)
        group_convs.append(pooled)
    
    # Concatenate the three resulting groups of feature maps
    concatenated_features = Concatenate(axis=-1)(group_convs)
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concatenated_features)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model