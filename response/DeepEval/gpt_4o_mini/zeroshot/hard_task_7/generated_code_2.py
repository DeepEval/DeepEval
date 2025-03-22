import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (28, 28, 1)  # MNIST image shape
    inputs = layers.Input(shape=input_shape)

    # Initial Convolution Layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1
    # Splitting the input into two groups
    split_outputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(x)
    group1 = split_outputs[0]
    group2 = split_outputs[1]

    # Operations on the first group
    group1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(group1)  # 1x1 Convolution
    group1 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(group1)  # Depthwise Separable Convolution
    group1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(group1)  # 1x1 Convolution

    # Merging the two groups
    merged = layers.Concatenate()([group1, group2])

    # Block 2
    # Reshape for channel shuffling
    height, width, channels = tf.shape(merged)[1], tf.shape(merged)[2], tf.shape(merged)[3]
    groups = 4
    channels_per_group = channels // groups

    reshaped = layers.Reshape((height, width, groups, channels_per_group))(merged)
    permuted = layers.Permute((1, 2, 4, 3))(reshaped)  # Swap groups and channels
    channel_shuffled = layers.Reshape((height, width, channels))(permuted)

    # Flattening and fully connected layer for classification
    x = layers.Flatten()(channel_shuffled)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# To create the model, you can call the function
model = dl_model()
model.summary()  # Display the model architecture