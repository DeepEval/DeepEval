import keras
from keras.layers import Input, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three channel groups
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    channel_groups = Lambda(split_channels)(input_layer)
    
    # Define the separable convolutional layers for each channel group
    conv1x1 = [SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(group) for group in channel_groups]
    conv3x3 = [SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(group) for group in channel_groups]
    conv5x5 = [SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(group) for group in channel_groups]
    
    # Concatenate outputs of separable convolutions
    concatenated = Concatenate()(conv1x1 + conv3x3 + conv5x5)
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model