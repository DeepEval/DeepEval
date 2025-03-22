import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel
    split_layer = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Define the first block
    block1_layer = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(split_layer[0])
    block1_layer = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(block1_layer)
    block1_layer = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(block1_layer)
    block1_layer = layers.Dropout(0.2)(block1_layer)
    block1_layer = layers.Concatenate()([split_layer[1], split_layer[2]])(block1_layer)

    # Define the second block
    block2_layer = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(block1_layer)
    block2_layer = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.MaxPooling2D((1, 1), strides=(1, 1), padding='same')(block2_layer)
    block2_layer = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(block2_layer)
    block2_layer = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(block2_layer)
    block2_layer = layers.MaxPooling2D((5, 5), strides=(1, 1), padding='same')(block2_layer)
    block2_layer = layers.Flatten()(block2_layer)
    block2_layer = layers.Dense(128, activation='relu')(block2_layer)
    block2_layer = layers.Dense(10, activation='softmax')(block2_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=block2_layer)

    return model