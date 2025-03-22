import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():     
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the three groups of convolutional layers
    group1 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    group1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(group1)
    group1 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group1)

    group2 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[1])
    group2 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(group2)
    group2 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group2)

    group3 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[2])
    group3 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(group3)
    group3 = layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group3)

    # Combine the outputs from the three groups using an addition operation
    main_path = layers.Add()([group1, group2, group3])

    # Fuse the main path with the original input layer through another addition
    combined_features = layers.Add()([main_path, input_layer])

    # Flatten the combined features into a one-dimensional vector
    flatten_layer = layers.Flatten()(combined_features)

    # Define the fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model