import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    fc1 = Dense(units=32, activation='relu')(global_avg_pool)
    fc2 = Dense(units=32, activation='relu')(fc1)
    
    # Reshape weights to align with input shape
    weights = Reshape((1, 1, 32))(fc2)
    
    # Multiply element-wise with input feature map
    multiplied = Multiply()([input_layer, weights])
    
    # Flatten the result
    flatten_layer = Flatten()(multiplied)
    
    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model