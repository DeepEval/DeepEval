import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block
    main_path = input_layer
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = layers.Dropout(0.2)(main_path)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.Dropout(0.2)(main_path)
    main_path = layers.Conv2D(128, (3, 3), activation='relu')(main_path)
    main_path = layers.Dropout(0.2)(main_path)
    main_path = layers.Flatten()(main_path)

    branch_path = input_layer
    branch_path = layers.Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = layers.Dropout(0.2)(branch_path)
    branch_path = layers.Conv2D(64, (3, 3), activation='relu')(branch_path)
    branch_path = layers.Dropout(0.2)(branch_path)
    branch_path = layers.Conv2D(128, (3, 3), activation='relu')(branch_path)
    branch_path = layers.Dropout(0.2)(branch_path)
    branch_path = layers.Flatten()(branch_path)

    # Second block
    input_layer = layers.Lambda(tf.split, output_shape=(3, 32, 32, 3))(input_layer)
    group1 = input_layer[0]
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)
    group1 = layers.Dropout(0.2)(group1)
    group1 = layers.Conv2D(64, (3, 3), activation='relu')(group1)
    group1 = layers.Dropout(0.2)(group1)
    group1 = layers.Flatten()(group1)
    group2 = input_layer[1]
    group2 = layers.Conv2D(32, (1, 1), activation='relu')(group2)
    group2 = layers.Dropout(0.2)(group2)
    group2 = layers.Conv2D(64, (3, 3), activation='relu')(group2)
    group2 = layers.Dropout(0.2)(group2)
    group2 = layers.Flatten()(group2)
    group3 = input_layer[2]
    group3 = layers.Conv2D(32, (1, 1), activation='relu')(group3)
    group3 = layers.Dropout(0.2)(group3)
    group3 = layers.Conv2D(64, (3, 3), activation='relu')(group3)
    group3 = layers.Dropout(0.2)(group3)
    group3 = layers.Flatten()(group3)

    # Concatenate features
    x = layers.Concatenate()([main_path, branch_path, group1, group2, group3])

    # Dense layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model