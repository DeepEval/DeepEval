import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     

    input_layer = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Apply 1x1 convolutions to each group independently
    conv_group1 = layers.Conv2D(32 // 3, (1, 1), padding='same')(split_layer[0])
    conv_group2 = layers.Conv2D(32 // 3, (1, 1), padding='same')(split_layer[1])
    conv_group3 = layers.Conv2D(32 // 3, (1, 1), padding='same')(split_layer[2])

    # Downsample each group via an average pooling layer
    avg_pool_group1 = layers.AveragePooling2D(pool_size=(2, 2))(conv_group1)
    avg_pool_group2 = layers.AveragePooling2D(pool_size=(2, 2))(conv_group2)
    avg_pool_group3 = layers.AveragePooling2D(pool_size=(2, 2))(conv_group3)

    # Concatenate the three groups along the channel dimension
    concat_layer = layers.Concatenate()([avg_pool_group1, avg_pool_group2, avg_pool_group3])

    # Flatten the concatenated feature maps into a one-dimensional vector
    flatten_layer = layers.Flatten()(concat_layer)

    # Apply two fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model