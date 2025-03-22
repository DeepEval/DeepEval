import tensorflow as tf
from tensorflow.keras import layers

def cbam_block(cbam_feature, ratio=8):
    x = layers.GlobalAveragePooling2D()(cbam_feature)
    x = layers.Dense(cbam_feature.shape[3] // ratio, activation='relu')(x)
    x = layers.Dense(cbam_feature.shape[3], activation='sigmoid')(x)

    # Apply the attention weights to the input feature map
    attention_map = layers.Reshape((1, 1, cbam_feature.shape[3]))(x)
    cbam_feature = layers.multiply([cbam_feature, attention_map])

    return cbam_feature

def se_block(se_feature, ratio=16):
    x = layers.GlobalAveragePooling2D()(se_feature)
    x = layers.Dense(se_feature.shape[3] // ratio, activation='relu')(x)
    x = layers.Dense(se_feature.shape[3], activation='sigmoid')(x)

    # Apply the squeeze-excitation weights to the input feature map
    se_feature = layers.multiply([se_feature, x])

    return se_feature

def block_1(x):
    # Path 1: Global average pooling followed by two fully connected layers
    path_1 = layers.GlobalAveragePooling2D()(x)
    path_1 = layers.Dense(x.shape[3] // 4, activation='relu')(path_1)
    path_1 = layers.Dense(x.shape[3], activation='sigmoid')(path_1)

    # Path 2: Global max pooling followed by two fully connected layers
    path_2 = layers.GlobalMaxPooling2D()(x)
    path_2 = layers.Dense(x.shape[3] // 4, activation='relu')(path_2)
    path_2 = layers.Dense(x.shape[3], activation='sigmoid')(path_2)

    # Add the outputs from both paths and apply an activation function
    path_combined = layers.Add()([path_1, path_2])
    path_combined = layers.Activation('relu')(path_combined)

    # Apply channel attention weights
    cbam_feature = cbam_block(path_combined)

    # Add the channel attention feature back to the original features
    path_final = layers.Add()([x, cbam_feature])

    return path_final

def block_2(x):
    # Average pooling branch
    avg_pool = layers.AveragePooling2D()(x)
    avg_pool = layers.Conv2D(x.shape[3] // 8, (1, 1), padding='same')(avg_pool)
    avg_pool = layers.Activation('relu')(avg_pool)

    # Max pooling branch
    max_pool = layers.MaxPooling2D()(x)
    max_pool = layers.Conv2D(x.shape[3] // 8, (1, 1), padding='same')(max_pool)
    max_pool = layers.Activation('relu')(max_pool)

    # Concatenate the outputs from both branches
    concat = layers.concatenate([avg_pool, max_pool], axis=3)

    # 1x1 convolution to normalize the features
    conv = layers.Conv2D(x.shape[3], (1, 1), padding='same')(concat)
    conv = layers.Activation('sigmoid')(conv)

    # Multiply the normalized features with the channel dimension features from Block 1
    spatial_features = layers.multiply([conv, block_1(x)])

    # Add the spatial features back to the original features
    path_final = layers.Add()([x, spatial_features])

    return path_final

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)

    # Block 1
    x = block_1(x)

    # Block 2
    x = block_2(x)

    # Final fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model