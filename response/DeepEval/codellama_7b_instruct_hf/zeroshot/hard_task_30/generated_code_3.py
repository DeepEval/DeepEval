import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = keras.layers.Input(shape=(32, 32, 3))

    # Block 1: Dual-path structure
    main_path = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = keras.layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = keras.layers.Conv2D(64, (3, 3), activation='relu')(main_path)

    branch_path = keras.layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch_path = keras.layers.Conv2D(64, (3, 3), activation='relu')(branch_path)

    combined_path = keras.layers.Add()([main_path, branch_path])

    # Block 2: Split and concatenate
    split_layers = [layers.Lambda(lambda x: tf.split(x, 3, axis=1))(combined_path) for i in range(3)]
    split_layers = [layers.Conv2D(64, (1, 1), activation='relu')(x) for x in split_layers]
    split_layers = [layers.Conv2D(64, (3, 3), activation='relu')(x) for x in split_layers]
    split_layers = [layers.Conv2D(64, (5, 5), activation='relu')(x) for x in split_layers]
    combined_path = layers.Concatenate()(split_layers)

    # Output layers
    output_layer = keras.layers.Flatten()(combined_path)
    output_layer = keras.layers.Dense(10, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model