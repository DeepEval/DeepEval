import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.layers import Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=64, activation='relu')(global_avg_pool)
    dense2_main = Dense(units=3, activation='relu')(dense1_main)  # CIFAR-10 has 3 channels in input
    weights_main = Reshape((1, 1, 3))(dense2_main)
    scaled_features = Multiply()([input_layer, weights_main])
    
    # Branch path (identity)
    branch_path = input_layer
    
    # Combine paths
    combined = Add()([scaled_features, branch_path])
    
    # Final fully connected layers
    flatten_layer = Flatten()(combined)
    dense1_output = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model