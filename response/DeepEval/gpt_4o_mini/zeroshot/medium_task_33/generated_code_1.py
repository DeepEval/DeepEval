import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three channel groups (R, G, B)
    red_channel, green_channel, blue_channel = tf.split(input_layer, num_or_size_splits=3, axis=-1)

    # Feature extraction for red channel using separable convolutions
    red_features = layers.Lambda(lambda x: x)(red_channel)  # Just passing through
    red_features = layers.SeparableConv2D(32, (1, 1), activation='relu')(red_features)
    red_features = layers.SeparableConv2D(64, (3, 3), activation='relu')(red_features)

    # Feature extraction for green channel using separable convolutions
    green_features = layers.Lambda(lambda x: x)(green_channel)  # Just passing through
    green_features = layers.SeparableConv2D(32, (3, 3), activation='relu')(green_features)
    green_features = layers.SeparableConv2D(64, (5, 5), activation='relu')(green_features)

    # Feature extraction for blue channel using separable convolutions
    blue_features = layers.Lambda(lambda x: x)(blue_channel)  # Just passing through
    blue_features = layers.SeparableConv2D(32, (5, 5), activation='relu')(blue_features)
    blue_features = layers.SeparableConv2D(64, (3, 3), activation='relu')(blue_features)

    # Concatenate the features from all three channels
    concatenated_features = layers.Concatenate()([red_features, green_features, blue_features])

    # Flatten the concatenated features
    flattened_features = layers.Flatten()(concatenated_features)

    # Fully connected layers
    dense_layer_1 = layers.Dense(128, activation='relu')(flattened_features)
    dense_layer_2 = layers.Dense(64, activation='relu')(dense_layer_1)
    dense_layer_3 = layers.Dense(10, activation='softmax')(dense_layer_2)  # Output layer for 10 classes

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=dense_layer_3)

    return model