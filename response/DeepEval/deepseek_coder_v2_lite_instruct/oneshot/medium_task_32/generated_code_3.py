import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the depthwise separable convolutional layers for each group
    def depthwise_separable_conv(input_tensor, kernel_size):
        conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', activation='relu', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(input_tensor)
        return conv
    
    # Apply depthwise separable convolutional layers with different kernel sizes
    conv1x1 = depthwise_separable_conv(split_layer[0], kernel_size=(1, 1))
    conv3x3 = depthwise_separable_conv(split_layer[1], kernel_size=(3, 3))
    conv5x5 = depthwise_separable_conv(split_layer[2], kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated features
    flattened = Flatten()(concatenated)
    
    # Pass the flattened features through a fully connected layer for classification
    dense = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model