import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer with shape (32, 32, 3) for CIFAR-10 dataset
    input_layer = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_input = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1, group2, group3 = split_input

    # Apply 1x1 convolutions to each group independently
    group1 = layers.Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same')(group1)
    group2 = layers.Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same')(group2)
    group3 = layers.Conv2D(filters=int(3/3), kernel_size=(1, 1), strides=(1, 1), padding='same')(group3)

    # Apply average pooling to each group
    group1 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group1)
    group2 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group2)
    group3 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(group3)

    # Concatenate the three groups along the channel dimension
    concatenated = layers.Concatenate()([group1, group2, group3])

    # Flatten the concatenated feature maps
    flattened = layers.Flatten()(concatenated)

    # Apply two fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flattened)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model