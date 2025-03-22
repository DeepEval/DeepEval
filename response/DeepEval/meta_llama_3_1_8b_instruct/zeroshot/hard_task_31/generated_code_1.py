# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def dl_model():
    # Define the input shape of the model
    input_shape = (32, 32, 3)

    # Define the model
    model = keras.Model()

    # Block 1
    input_layer = keras.Input(shape=input_shape)

    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    main_path = layers.Dropout(0.2)(main_path)
    main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = layers.Flatten()(main_path)

    # Branch path
    branch_path = input_layer

    # Merge the two paths
    merged_path = layers.Add()([main_path, branch_path])

    # Block 2
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(merged_path)
    x = layers.SeparableConv2D(32, (1, 1), padding='same')(x[0])
    x = layers.SeparableConv2D(32, (3, 3), padding='same')(x[1])
    x = layers.SeparableConv2D(32, (5, 5), padding='same')(x[2])
    x = layers.Concatenate(axis=-1)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Combine the layers into the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])