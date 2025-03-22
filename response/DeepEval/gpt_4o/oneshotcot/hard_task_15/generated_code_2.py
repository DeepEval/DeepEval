from keras import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense_main_1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense_main_2 = Dense(units=3, activation='relu')(dense_main_1)  # CIFAR-10 has 3 channels

    # Reshape to match the input layer's shape for element-wise multiplication
    reshaped_weights = keras.layers.Reshape((1, 1, 3))(dense_main_2)
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Branch Path
    branch_output = input_layer  # Direct connection

    # Add outputs from both paths
    added_output = Add()([weighted_input, branch_output])

    # Final Fully Connected Layers
    dense_final_1 = Dense(units=128, activation='relu')(added_output)
    dense_final_2 = Dense(units=10, activation='softmax')(dense_final_1)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=dense_final_2)

    return model