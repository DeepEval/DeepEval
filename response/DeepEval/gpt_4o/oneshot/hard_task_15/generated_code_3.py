import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten, Activation

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path with Global Average Pooling and fully connected layers
    global_avg_pooling = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=64, activation='relu')(global_avg_pooling)
    dense2_main = Dense(units=3, activation='sigmoid')(dense1_main)  # same number of channels as input layer
    weights = Reshape((1, 1, 3))(dense2_main)  # Reshape to match input shape
    scaled_features = Multiply()([input_layer, weights])  # Element-wise multiplication with input

    # Branch path directly from input
    branch_path = input_layer  # Directly using the input layer

    # Combine both paths
    combined = Add()([scaled_features, branch_path])

    # Final fully connected layers
    flatten_combined = Flatten()(combined)
    dense1_final = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(dense1_final)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model