import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    # Define the input layer with shape suitable for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input along the channel dimension into three equal groups
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv_group1 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_channels[0])
    conv_group2 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_channels[1])
    conv_group3 = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(split_channels[2])
    
    # Downsample each group using average pooling
    avg_pool_group1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group1)
    avg_pool_group2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group2)
    avg_pool_group3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_group3)
    
    # Concatenate the pooled feature maps along the channel dimension
    concatenated_features = Concatenate(axis=-1)([avg_pool_group1, avg_pool_group2, avg_pool_group3])
    
    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concatenated_features)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model