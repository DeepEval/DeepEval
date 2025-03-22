import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(28, 28, 1))

    # First Block
    x = layers.AvgPool2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
    x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.AvgPool2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = layers.Flatten()(x) 
    
    # Concatenate outputs from pooling layers
    x = layers.Concatenate()([x, x, x]) 

    # Fully connected layer and reshape
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 128, 1))(x)

    # Second Block
    x = layers.Lambda(lambda x: tf.split(x, 4, axis=2))(x)
    
    # Depthwise Separable Convolutions
    branches = []
    for kernel_size in [1, 3, 5, 7]:
        branch = layers.DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(x)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Activation('relu')(branch)
        branches.append(branch)

    # Concatenate outputs from branches
    x = layers.Concatenate(axis=2)(branches)
    x = layers.Flatten()(x)
    
    # Final Classification Layer
    output = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output)
    
    return model