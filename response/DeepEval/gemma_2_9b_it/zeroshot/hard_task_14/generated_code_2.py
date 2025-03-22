import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    main_branch_output = layers.Reshape((32, 32, 32))(x)  

    # Branch Path
    branch_x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch_x = layers.Conv2D(32, (3, 3), activation='relu')(branch_x)
    branch_output = layers.Conv2D(32, (3, 3), activation='relu')(branch_x)

    # Add outputs
    combined_output = layers.Add()([main_branch_output, branch_output])

    # Final Classification Layers
    x = layers.Flatten()(combined_output)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model