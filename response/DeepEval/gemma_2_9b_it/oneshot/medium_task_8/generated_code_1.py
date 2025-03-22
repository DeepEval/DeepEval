import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    input_tensor = layers.Input(shape=(32, 32, 3))

    # Main Path
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
    
    # Group 1 (unchanged)
    group1 = split_tensor[0]

    # Group 2 (feature extraction)
    group2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(split_tensor[1])

    # Group 3 (combined with group 2)
    group3 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tf.concat([split_tensor[2], group2], axis=-1))

    # Concatenate outputs of all groups
    main_path_output = layers.Concatenate()([group1, group3])

    # Branch Path
    branch_path_output = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)

    # Fuse outputs
    fused_output = layers.Add()([main_path_output, branch_path_output])

    # Flatten and classify
    flatten_layer = layers.Flatten()(fused_output)
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)

    return model