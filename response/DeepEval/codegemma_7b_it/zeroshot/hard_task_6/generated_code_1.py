import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Block 1
    splitted_inputs = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    block1_outputs = []
    for split_input in splitted_inputs:
        x = layers.Conv2D(filters=64, kernel_size=1, activation='relu')(split_input)
        block1_outputs.append(x)
    fused_features_block1 = layers.concatenate(block1_outputs)

    # Branch path
    branch_features = layers.AveragePooling2D(pool_size=8)(inputs)
    branch_features = layers.Flatten()(branch_features)

    # Block 2
    feature_shape = tf.keras.backend.int_shape(fused_features_block1)
    reshape_features = layers.Reshape((feature_shape[1], feature_shape[2], 3, feature_shape[3] // 3))(fused_features_block1)
    permutated_features = layers.Permute((2, 3, 1, 4))(reshape_features)
    reshaped_features = layers.Reshape((feature_shape[1], feature_shape[2], feature_shape[3]))(permutated_features)

    # Block 3
    block3_features = layers.DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(reshaped_features)

    # Concatenate outputs from main path and branch path
    concat_features = layers.concatenate([block3_features, branch_features])

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(concat_features)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model