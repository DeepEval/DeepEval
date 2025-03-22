import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Reshape and multiply with input feature map
    weights = Reshape((32, 32, 3))(fc2)
    weighted_input = Multiply()([input_layer, weights])
    
    # Flatten the result
    flatten_layer = Flatten()(weighted_input)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model