import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_img = keras.Input(shape=(32, 32, 3), name='input_image')

    # Branch 1: 3x3 kernel
    branch_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    branch_1 = layers.BatchNormalization()(branch_1)

    # Branch 2: 5x5 kernel
    branch_2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(input_img)
    branch_2 = layers.BatchNormalization()(branch_2)

    # Combine branches through addition
    concat = layers.Add()([branch_1, branch_2])

    # Global average pooling
    avg_pool = layers.GlobalAveragePooling2D()(concat)

    # Fully connected attention layers
    attention_fc1 = layers.Dense(32, activation='relu')(avg_pool)
    attention_fc1 = layers.BatchNormalization()(attention_fc1)
    attention_fc2 = layers.Dense(10, activation='softmax', name='attention')(attention_fc1)

    # Weighted output
    weighted_output = layers.multiply([concat, attention_fc2])
    weighted_output = layers.Add()([weighted_output, branch_2])

    # Final fully connected layer
    output = layers.Dense(10, activation='softmax', name='output')(weighted_output)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model