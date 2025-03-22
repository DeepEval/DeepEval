import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     

    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    group1, group2, group3 = split_layer

    # Apply 1x1 convolution to each group
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv2 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group2)
    conv3 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group3)

    # Downsample each group via average pooling layer
    avg_pool1 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    avg_pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    avg_pool3 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Concatenate the three groups along the channel dimension
    concatenated = layers.Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    # Flatten the concatenated feature maps into a one-dimensional vector
    flatten = layers.Flatten()(concatenated)

    # Apply two fully connected layers for classification
    dense1 = layers.Dense(units=128, activation='relu')(flatten)
    dense2 = layers.Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model