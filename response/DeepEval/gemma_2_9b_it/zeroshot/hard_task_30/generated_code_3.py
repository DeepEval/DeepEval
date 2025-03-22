import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1: Dual-path Structure
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.Conv2D(3, (3, 3), activation='relu')(main_path)

    branch_path = layers.Conv2D(3, (1, 1), activation='relu')(input_layer)

    # Combine paths using addition
    x = layers.Add()([main_path, branch_path])

    # Block 2: Channel Splitting and Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)  
    
    # Group 1: 1x1 kernel
    group1_output = layers.Conv2D(16, (1, 1), activation='relu', depth_wise=True)(x[0])
    group1_output = layers.BatchNormalization()(group1_output)

    # Group 2: 3x3 kernel
    group2_output = layers.Conv2D(32, (3, 3), activation='relu', depth_wise=True)(x[1])
    group2_output = layers.BatchNormalization()(group2_output)

    # Group 3: 5x5 kernel
    group3_output = layers.Conv2D(64, (5, 5), activation='relu', depth_wise=True)(x[2])
    group3_output = layers.BatchNormalization()(group3_output)

    # Concatenate outputs from each group
    x = layers.Concatenate()([group1_output, group2_output, group3_output])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model