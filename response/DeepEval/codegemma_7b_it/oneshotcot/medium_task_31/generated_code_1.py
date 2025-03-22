import keras
from keras import layers
from tensorflow.keras.models import Model

def dl_model():

    # Input layer for CIFAR-10 images
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the image into three groups along the channel dimension
    group1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    group2 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    group3 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply different convolutional kernels to each group
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(group1)
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(group2)
    conv3 = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(group3)

    # Concatenate the outputs from the three groups
    concat = layers.Concatenate()([conv1, conv2, conv3])

    # Flatten the concatenated features
    flatten = layers.Flatten()(concat)

    # Fully connected layers for classification
    dense1 = layers.Dense(units=128, activation='relu')(flatten)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model