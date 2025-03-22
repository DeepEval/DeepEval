import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = tf.keras.Input(shape=(32, 32, 3))

    # First Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D((2, 2))(x)

    # Second Block
    main_path = layers.GlobalAveragePooling2D()(x)
    main_path = layers.Dense(64, activation='relu')(main_path)
    main_path = layers.Dense(64, activation='relu')(main_path)

    # Channel Weight Refinement
    channel_weights = layers.Reshape((64, 64))(main_path)
    channel_weights = layers.multiply([channel_weights, x])

    # Flatten and Classify
    x = layers.Flatten()(channel_weights)
    output = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    
    return model