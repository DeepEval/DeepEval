import keras
from keras.layers import Input, GlobalAveragePooling2D, Multiply, Dense, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # We use sigmoid to ensure the weights are between 0 and 1
    reshaped_weights = keras.layers.Reshape(target_shape=(1, 1, 3))(dense2)
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Branch path (direct connection)
    branch_path = input_layer

    # Combine paths
    combined_output = Add()([weighted_input, branch_path])

    # Final fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model