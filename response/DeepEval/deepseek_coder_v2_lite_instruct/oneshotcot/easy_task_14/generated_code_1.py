import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape weights to align with the input shape
    weights = Reshape((1, 1, 64))(dense2)
    
    # Multiply element-wise with the input feature map
    multiplied = Multiply()([input_layer, weights])
    
    # Flatten the result
    flattened = Flatten()(multiplied)
    
    # Another fully connected layer to obtain the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model