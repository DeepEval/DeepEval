import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    input_img = keras.Input(shape=(32, 32, 3))

    # First block
    branch_1x1 = layers.Conv2D(64, (1, 1), padding='same')(input_img)
    branch_3x3 = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    branch_5x5 = layers.Conv2D(64, (5, 5), padding='same')(input_img)
    branch_max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)

    # Concatenate branches
    concat_layer = layers.concatenate([branch_1x1, branch_3x3, branch_5x5, branch_max_pool])

    # Second block
    x = layers.BatchNormalization()(concat_layer)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Reshape weights to match input shape
    weights = layers.Reshape((64, 32, 32))(x)

    # Element-wise multiplication with input feature map
    output = layers.Multiply()([weights, concat_layer])

    # Final fully connected layer
    output = layers.Conv2D(10, (1, 1), padding='same')(output)

    # Output probability distribution
    output = layers.Activation('softmax')(output)

    # Create the model
    model = keras.Model(inputs=input_img, outputs=output)

    return model