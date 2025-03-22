import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer for the original image
    input_image = layers.Input(shape=(32, 32, 3))

    # Split the image into three groups along the channel dimension
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=3))
    group_1, group_2, group_3 = split_layer(input_image)

    # Extract features from the first group
    group_1_conv_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group_1)
    group_1_conv_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group_1_conv_1)
    group_1_conv_3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group_1_conv_2)

    # Extract features from the second group
    group_2_conv_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group_2)
    group_2_conv_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group_2_conv_1)
    group_2_conv_3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group_2_conv_2)

    # Extract features from the third group
    group_3_conv_1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group_3)
    group_3_conv_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(group_3_conv_1)
    group_3_conv_3 = layers.Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(group_3_conv_2)

    # Combine the outputs from the three groups
    concat_layer = layers.concatenate([group_1_conv_3, group_2_conv_3, group_3_conv_3])

    # Fuse the main path with the original input layer
    fusion_layer = layers.add([concat_layer, input_image])

    # Flatten the combined features
    flatten_layer = layers.Flatten()(fusion_layer)

    # Fully connected layer for classification
    output_layer = layers.Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_image, outputs=output_layer)

    return model