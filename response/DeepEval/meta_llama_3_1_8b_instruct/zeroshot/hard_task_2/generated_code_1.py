import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    group_1 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[0])(inputs)
    group_2 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[1])(inputs)
    group_3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1)[2])(inputs)

    # Define the convolutional layers for each group
    conv_1_group_1 = layers.Conv2D(32, (3, 3), activation='relu')(group_1)
    conv_2_group_1 = layers.Conv2D(32, (3, 3), activation='relu')(conv_1_group_1)
    conv_3_group_1 = layers.Conv2D(32, (3, 3), activation='relu')(conv_2_group_1)

    conv_1_group_2 = layers.Conv2D(32, (3, 3), activation='relu')(group_2)
    conv_2_group_2 = layers.Conv2D(32, (3, 3), activation='relu')(conv_1_group_2)
    conv_3_group_2 = layers.Conv2D(32, (3, 3), activation='relu')(conv_2_group_2)

    conv_1_group_3 = layers.Conv2D(32, (3, 3), activation='relu')(group_3)
    conv_2_group_3 = layers.Conv2D(32, (3, 3), activation='relu')(conv_1_group_3)
    conv_3_group_3 = layers.Conv2D(32, (3, 3), activation='relu')(conv_2_group_3)

    # Combine the outputs from the three groups
    combined_group_output = layers.Add()([conv_3_group_1, conv_3_group_2, conv_3_group_3])

    # Fuse the combined features with the original input layer
    combined_input_output = layers.Add()([inputs, combined_group_output])

    # Flatten the combined features
    flattened_output = layers.Flatten()(combined_input_output)

    # Define the fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model