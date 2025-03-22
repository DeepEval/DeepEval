import keras
from keras import layers

def dl_model():

    inputs = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    group1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    group2 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    group3 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)

    # Apply different convolutional kernels to each group
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group1)
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(group2)
    conv3 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(group3)

    # Concatenate the outputs from the three groups
    concat = layers.Concatenate(axis=3)([conv1, conv2, conv3])

    # Flatten the concatenated features
    flatten = layers.Flatten()(concat)

    # Fully connected layers for classification
    dense1 = layers.Dense(units=128, activation='relu')(flatten)
    outputs = layers.Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model