import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Fully Connected Layers to generate weights
    fc1 = Dense(units=32, activation='relu')(gap)
    weights = Dense(units=32*32*3, activation='sigmoid')(fc1)
    
    # Reshape weights to align with input shape
    weights_reshaped = tf.reshape(weights, (32, 32, 3))
    
    # Element-wise multiplication with input feature map
    multiplied = Multiply()([input_layer, weights_reshaped])
    
    # Flatten the result
    flattened = Flatten()(multiplied)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model