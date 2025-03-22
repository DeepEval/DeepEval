import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2_main = Dense(units=3, activation='relu')(dense1_main)  # Assuming the 'same number of channels' means 3 channels
    weights = Flatten()(dense2_main)  # Flatten to reshape later
    reshaped_weights = keras.layers.Reshape((1, 1, 3))(weights)  # Reshape to match input layer's channels
    scaled_features = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Branch path
    branch_path = input_layer  # Direct connection to the input layer
    
    # Add main and branch paths
    combined_output = Add()([scaled_features, branch_path])
    
    # Final fully connected layers
    dense3 = Dense(units=64, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model