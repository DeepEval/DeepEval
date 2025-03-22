import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from tensorflow.keras.applications import ResNet50

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolution to increase the dimensionality
    x = Conv2D(3, (1, 1), padding='same', activation='relu')(inputs)
    
    # Depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Channel attention mechanism
    # Global average pooling
    gap = GlobalAveragePooling2D()(x)
    
    # Two fully connected layers for generating weights
    fc1 = Dense(x.shape[-1] // 4, activation='relu')(gap)
    fc2 = Dense(x.shape[-1], activation='sigmoid')(fc1)
    
    # Reshape to match the initial features
    fc2 = tf.reshape(fc2, (-1, 1, 1, x.shape[-1]))
    
    # Multiply the initial features with the weights
    x = Multiply()([x, fc2])
    
    # 1x1 convolution to reduce dimensionality
    x = Conv2D(3, (1, 1), padding='same', activation='relu')(x)
    
    # Add the output to the initial input
    x = Add()([x, inputs])
    
    # Flatten layer
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()