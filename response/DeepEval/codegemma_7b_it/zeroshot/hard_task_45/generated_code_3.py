import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the model architecture
def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Feature extraction with depthwise separable convolutions
    split_input = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    block1_outputs = []
    for i in range(3):
        x = split_input[i]
        x = layers.SeparableConv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(16, (5, 5), padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        block1_outputs.append(x)

    # Concatenate the outputs from Block 1
    block1_output = layers.Concatenate(axis=-1)(block1_outputs)

    # Block 2: Feature extraction with multiple branches
    branch1 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(block1_output)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Activation('relu')(branch1)

    branch2 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(block1_output)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Activation('relu')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(branch2)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Activation('relu')(branch2)

    branch3 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(block1_output)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.Activation('relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(branch3)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.Activation('relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(branch3)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.Activation('relu')(branch3)

    branch4 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(block1_output)
    branch4 = layers.Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(branch4)
    branch4 = layers.BatchNormalization()(branch4)
    branch4 = layers.Activation('relu')(branch4)

    # Concatenate the outputs from all branches
    concat_branches = layers.Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Feature extraction and classification
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(concat_branches)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model