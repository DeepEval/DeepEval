import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1
    main_path = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(input_layer)
    main_path = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(main_path)
    main_path = layers.Conv2D(filters=3, kernel_size=1)(main_path)

    branch_path = layers.Conv2D(filters=3, kernel_size=1)(input_layer)

    combined_output = layers.add([main_path, branch_path])

    # Block 2
    split_output = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(combined_output)
    
    group1 = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(split_output[0])
    group2 = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(split_output[1])
    group3 = layers.Conv2D(filters=64, kernel_size=5, activation='relu')(split_output[2])

    concatenated_output = layers.concatenate([group1, group2, group3], axis=3)

    # Fully Connected Layers
    flattened_output = layers.Flatten()(concatenated_output)
    dense1 = layers.Dense(128, activation='relu')(flattened_output)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model