import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    # Define input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three channel groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    channel_groups = Lambda(split_channels)(input_layer)
    
    # Apply separable convolutions to each channel group
    def separable_conv_block(channel):
        path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channel)
        path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(channel)
        path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(channel)
        return Concatenate()([path1, path2, path3])
    
    processed_channels = [separable_conv_block(channel) for channel in channel_groups]
    
    # Concatenate the processed channel outputs
    concatenated_features = Concatenate()(processed_channels)
    
    # Flatten the feature map
    flatten_layer = Flatten()(concatenated_features)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model