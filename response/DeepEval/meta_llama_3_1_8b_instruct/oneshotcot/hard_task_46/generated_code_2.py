import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, GlobalAveragePooling2D, Dense

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Three branches of separable convolution
    def separable_conv_branch(kernel_size, filters):
        x = layers.SeparableConv2D(filters, (kernel_size, kernel_size), strides=1, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    branch1 = separable_conv_branch(1, 64)
    branch2 = separable_conv_branch(3, 64)
    branch3 = separable_conv_branch(5, 64)
    
    # Concatenate the outputs of the three branches
    merged = Concatenate()([branch1, branch2, branch3])
    
    # Block 2: Multiple branches for enhanced feature extraction
    def convolution_branch(kernel_size, filters):
        x = layers.Conv2D(filters, (kernel_size, kernel_size), strides=1, padding='same')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    branch4 = convolution_branch(3, 128)
    branch5 = layers.Conv2D(128, (1, 1), strides=1, padding='same')(merged)
    branch5 = convolution_branch(3, 128)(branch5)
    branch5 = convolution_branch(3, 128)(branch5)
    
    # Max pooling branch
    branch6 = layers.MaxPooling2D((2, 2))(merged)
    
    # Concatenate the outputs of all branches
    merged2 = Concatenate()([branch4, branch5, branch6])
    
    # Global average pooling
    global_pool = GlobalAveragePooling2D()(merged2)
    
    # Output layer
    output_layer = Dense(10, activation='softmax')(global_pool)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model