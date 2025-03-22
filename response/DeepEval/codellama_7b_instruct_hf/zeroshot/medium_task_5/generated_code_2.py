import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)

    # Branch path
    branch_path = layers.Conv2D(16, (3, 3), activation='relu')(input_layer)
    branch_path = layers.MaxPooling2D((2, 2))(branch_path)

    # Combine main and branch paths
    combined_path = layers.Add()([main_path, branch_path])

    # Flatten and project onto probability distribution
    flattened_path = layers.Flatten()(combined_path)
    projected_path = layers.Dense(10, activation='softmax')(flattened_path)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=projected_path)
    return model