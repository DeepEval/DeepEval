import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the first block
    input_tensor = keras.Input(shape=(32, 32, 3))  

    # Main path
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(3, (3, 3), padding='same', activation='relu')(x)

    # Branch path
    branch_x = layers.Conv2D(3, (1, 1), padding='same', activation='relu')(input_tensor)

    # Add outputs from main and branch paths
    x = layers.Add()([x, branch_x])

    # Define the second block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)  # Split input into 3 groups
    
    # Group 1
    group1_x = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x[0])
    group1_x = layers.Dropout(0.2)(group1_x)
    
    # Group 2
    group2_x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x[1])
    group2_x = layers.Dropout(0.2)(group2_x)
    
    # Group 3
    group3_x = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(x[2])
    group3_x = layers.Dropout(0.2)(group3_x)

    x = layers.Concatenate()([group1_x, group2_x, group3_x])  # Concatenate outputs

    # Flatten and output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=input_tensor, outputs=outputs)
    return model