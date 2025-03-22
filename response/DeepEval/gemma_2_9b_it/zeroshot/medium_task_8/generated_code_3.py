import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))  

    # Main Path
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    
    # Group 1 remains unchanged
    branch1 = x[0]

    # Group 2 feature extraction
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(x[1])

    # Combine Group 2 and 3
    combined = layers.Conv2D(64, (3, 3), activation='relu')(tf.concat([branch2, x[2]], axis=3))

    # Concatenate all branches
    main_path_output = layers.concatenate([branch1, combined], axis=3)

    # Branch Path
    branch_path_output = layers.Conv2D(128, (1, 1), activation='relu')(inputs)

    # Fusion
    x = layers.add([main_path_output, branch_path_output])

    # Classification
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model