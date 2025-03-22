import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Define the main path
    main_input = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(main_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the branch path
    branch_input = keras.Input(shape=(32, 32, 3))
    branch_x = layers.Conv2D(32, (3, 3), activation='relu')(branch_input)
    branch_x = layers.MaxPooling2D()(branch_x)
    branch_x = layers.Conv2D(64, (3, 3), activation='relu')(branch_x)
    branch_x = layers.MaxPooling2D()(branch_x)
    branch_x = layers.Conv2D(128, (3, 3), activation='relu')(branch_x)
    branch_x = layers.MaxPooling2D()(branch_x)
    branch_x = layers.Dense(128, activation='relu')(branch_x)

    # Combine the outputs from the main and branch paths
    combined = layers.add([x, branch_x])
    combined = layers.Dense(64, activation='relu')(combined)
    combined = layers.Dense(10, activation='softmax')(combined)

    # Create the model
    model = keras.Model(inputs=[main_input, branch_input], outputs=combined)

    return model