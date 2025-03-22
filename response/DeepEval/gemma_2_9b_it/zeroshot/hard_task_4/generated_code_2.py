import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    inputs = keras.Input(shape=(32, 32, 3)) 
    
    # Increase channel dimensionality threefold
    x = layers.Conv2D(filters=9, kernel_size=(1, 1), activation='relu')(inputs) 
    
    # Depthwise separable convolution for feature extraction
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Channel attention module
    global_average_pool = layers.GlobalAveragePooling2D()(x)
    attention_weights = layers.Dense(units=9, activation='sigmoid')(global_average_pool)
    attention_weights = layers.Reshape((1, 1, 9))(attention_weights)
    x = layers.Multiply()([x, attention_weights])

    # Reduce dimensionality
    x = layers.Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(x) 

    # Element-wise summation with input
    x = layers.Add()([inputs, x]) 

    # Flatten and fully connected layers for classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model