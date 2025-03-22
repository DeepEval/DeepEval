import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    
    # Fully connected layers to generate weights
    fc1 = Dense(units=input_layer.shape[3] * 4, activation='relu')(global_avg_pool)
    fc2 = Dense(units=input_layer.shape[3] * input_layer.shape[3], activation='relu')(fc1)
    
    # Reshape weights to align with input shape
    reshape_weights = Reshape((input_layer.shape[3], input_layer.shape[3], 1))(fc2)
    
    # Multiply weights with input feature map
    weighted_input = Multiply()([input_layer, reshape_weights])
    
    # Flatten the result
    flatten_layer = Flatten()(weighted_input)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model