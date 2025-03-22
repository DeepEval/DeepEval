import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def depthwise_separable_conv(input_tensor, kernel_size):
        # Depthwise convolution
        depthwise_conv = Conv2D(filters=None, kernel_size=kernel_size, padding='same', depthwise_initializer='he_normal', depthwise_regularizer=keras.regularizers.l2(1e-4), activation='relu')(input_tensor)
        # Pointwise convolution
        pointwise_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(depthwise_conv)
        return pointwise_conv
    
    # Apply depthwise separable convolution to each group with different kernel sizes
    conv1 = depthwise_separable_conv(split_layer[0], kernel_size=(1, 1))
    conv2 = depthwise_separable_conv(split_layer[1], kernel_size=(3, 3))
    conv3 = depthwise_separable_conv(split_layer[2], kernel_size=(5, 5))
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])
    
    # Flatten the fused features
    flattened = Flatten()(concatenated)
    
    # Fully connected layer for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()