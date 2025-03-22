import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Global Average Pooling and two fully connected layers
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=64, activation='relu')(global_avg_pool)
    dense2_main = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1_main)  # Match the input layer's number of elements
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(dense2_main)
    weighted_feature_map = Multiply()([input_layer, reshaped_weights])

    # Branch path: Directly connected to the input layer
    branch_path = input_layer

    # Combine both paths using addition
    combined_output = Add()([weighted_feature_map, branch_path])

    # Final fully connected layers after combination
    flatten = Flatten()(combined_output)
    dense1_final = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1_final)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model