import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    
    # Group 1
    branch1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[0])
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch1)
    
    # Group 2
    branch2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[1])
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch2)
    
    # Group 3
    branch3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x[2])
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(branch3)
    
    main_path_output = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Branch Path
    branch_output = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(input_tensor)

    # Fusion
    fused_output = layers.add([main_path_output, branch_output])

    # Classification Layers
    x = layers.Flatten()(fused_output)
    x = layers.Dense(128, activation='relu')(x)
    output_tensor = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model