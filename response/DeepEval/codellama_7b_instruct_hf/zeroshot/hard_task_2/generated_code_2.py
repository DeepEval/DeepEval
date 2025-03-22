import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Splits the input into three groups along the channel dimension
    groups = tf.split(input_layer, num_or_size_splits=3, axis=3)

    # First group
    group1 = groups[0]
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)
    group1 = layers.Conv2D(64, (3, 3), activation='relu')(group1)
    group1 = layers.Conv2D(32, (1, 1), activation='relu')(group1)

    # Second group
    group2 = groups[1]
    group2 = layers.Conv2D(32, (1, 1), activation='relu')(group2)
    group2 = layers.Conv2D(64, (3, 3), activation='relu')(group2)
    group2 = layers.Conv2D(32, (1, 1), activation='relu')(group2)

    # Third group
    group3 = groups[2]
    group3 = layers.Conv2D(32, (1, 1), activation='relu')(group3)
    group3 = layers.Conv2D(64, (3, 3), activation='relu')(group3)
    group3 = layers.Conv2D(32, (1, 1), activation='relu')(group3)

    # Combine the outputs of the three groups using an addition operation
    main_path = layers.Add()([group1, group2, group3])

    # Fuse the main path with the original input layer through another addition
    output_layer = layers.Add()([main_path, input_layer])

    # Flatten the output of the addition layer and feed it into a fully connected layer for classification
    flattened_output = layers.Flatten()(output_layer)
    classification_layer = layers.Dense(10, activation='softmax')(flattened_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=classification_layer)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model