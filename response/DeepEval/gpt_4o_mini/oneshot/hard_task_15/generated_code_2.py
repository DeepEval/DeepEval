import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=128, activation='relu')(main_path)
    dense2_main = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1_main)  # Output shape matches the input

    # Reshape to match input layer shape
    reshaped_weights = keras.layers.Reshape((32, 32, 3))(dense2_main)
    weighted_features = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication with input features

    # Branch path
    branch_path = input_layer  # Directly connected to the input layer

    # Combine both paths
    combined_output = Add()([weighted_features, branch_path])

    # Pass through two fully connected layers
    flatten_combined = Flatten()(combined_output)
    dense1_combined = Dense(units=128, activation='relu')(flatten_combined)
    output_layer = Dense(units=10, activation='softmax')(dense1_combined)  # CIFAR-10 has 10 classes

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model