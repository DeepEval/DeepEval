import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    # Input layer with CIFAR-10 image size and 3 color channels
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input channels into three groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Applying 1x1 Convolutions to each group independently
    conv_group1 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[0])
    conv_group2 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[1])
    conv_group3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_channels[2])
    
    # Average pooling for downsampling each group
    avg_pool_group1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group1)
    avg_pool_group2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group2)
    avg_pool_group3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group3)
    
    # Concatenating the resulting feature maps along the channel dimension
    concatenated_features = Concatenate(axis=-1)([avg_pool_group1, avg_pool_group2, avg_pool_group3])
    
    # Flattening the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated_features)
    
    # Passing through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model