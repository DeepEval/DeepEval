import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate weights
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Reshape weights to align with input shape
    weights = Reshape((1, 1, 64))(fc2)
    
    # Multiply element-wise with the input feature map
    weighted_input = Multiply()([input_layer, weights])
    
    # Flatten the result
    flatten_layer = Flatten()(weighted_input)
    
    # Fully connected layer to obtain final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model