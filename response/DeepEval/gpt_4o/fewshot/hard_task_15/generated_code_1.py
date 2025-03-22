import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.datasets import cifar10

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32 * 3, activation='relu')(global_avg_pool)  # assuming 32 as a placeholder, adjust as necessary
    dense2 = Dense(units=32 * 3, activation='relu')(dense1)
    weights = Reshape(target_shape=(1, 1, 32 * 3))(dense2)
    scaled_features = Multiply()([input_layer, weights])
    
    # Branch path
    branch_path = input_layer  # direct connection
    
    # Combine paths
    combined = Add()([scaled_features, branch_path])
    
    # Fully connected layers for final classification
    flatten = Flatten()(combined)
    dense3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model