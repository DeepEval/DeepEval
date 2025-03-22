# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape of the model
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # First block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)

    # Branch path for the first block
    branch_x = layers.Lambda(lambda x: x)(inputs)

    # Add outputs from main path and branch path
    x = layers.Add()([x, branch_x])

    # Second block
    # Split the input into three groups along the last dimension
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)

    # Define the separable convolutional layers for each group
    group1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(x[0])
    group1 = layers.Dropout(0.2)(group1)

    group2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(x[1])
    group2 = layers.Dropout(0.2)(group2)

    group3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(x[2])
    group3 = layers.Dropout(0.2)(group3)

    # Concatenate the outputs from the three groups
    x = layers.Concatenate()([group1, group2, group3])

    # Flatten the output
    x = layers.Flatten()(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()