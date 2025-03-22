import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the depthwise separable convolutional layers with different kernel sizes
    def depthwise_separable_conv(input_tensor, kernel_size):
        conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', activation='relu', depthwise_kernel_initializer='he_normal')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv)
        return conv
    
    # Process each group with depthwise separable convolutional layers
    group1 = depthwise_separable_conv(split_layer[0], kernel_size=(1, 1))
    group2 = depthwise_separable_conv(split_layer[1], kernel_size=(3, 3))
    group3 = depthwise_separable_conv(split_layer[2], kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([group1, group2, group3])
    
    # Flatten the result
    flatten_layer = Flatten()(concatenated)
    
    # Pass through a fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model