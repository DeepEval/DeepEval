import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))
    branch_input = layers.Input(shape=(32, 32, 3))

    # Block 1
    conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat = layers.Concatenate(axis=3)([conv1, conv2, conv3])
    branch = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)
    merged = layers.Add()([concat, branch])
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(merged)

    # Block 2
    conv4 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv5 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv6 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool1)
    concat2 = layers.Concatenate(axis=3)([conv4, conv5, conv6])
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concat2)

    # Block 3
    conv7 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv8 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv9 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool2)
    concat3 = layers.Concatenate(axis=3)([conv7, conv8, conv9])
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concat3)

    # Final block
    conv10 = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool3)
    conv11 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    conv12 = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool3)
    concat4 = layers.Concatenate(axis=3)([conv10, conv11, conv12])
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(concat4)

    # Classification block
    flatten = layers.Flatten()(pool4)
    dense = layers.Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=[input_layer, branch_input], outputs=dense)
    return model