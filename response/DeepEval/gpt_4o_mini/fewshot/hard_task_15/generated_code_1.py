import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=128, activation='relu')(main_path)
    dense2_main = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1_main)  # Output same shape as input channels
    reshaped_weights = Dense(units=3 * 32 * 32)(dense2_main)  # Reshape to match input layer
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(reshaped_weights)  # Reshape to (32, 32, 3)
    weighted_input = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Branch path
    branch_path = input_layer  # Direct connection to input layer

    # Combine both paths
    combined_output = Add()([weighted_input, branch_path])

    # Feed the combined output through fully connected layers
    flatten_combined = Flatten()(combined_output)
    dense1_combined = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(dense1_combined)  # 10 classes for CIFAR-10

    model = Model(inputs=input_layer, outputs=output_layer)

    return model