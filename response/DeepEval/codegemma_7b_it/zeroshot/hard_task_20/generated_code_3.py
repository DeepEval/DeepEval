import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_image = keras.Input(shape=(32, 32, 3))

    # Main Path
    main_path = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_image)
    main_path = [
        layers.Conv2D(64, (1, 1), padding="same", activation="relu")(x)
        for x in main_path
    ]
    main_path = [
        layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        for x in main_path
    ]
    main_path = [
        layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x) for x in main_path
    ]

    # Branch Path
    branch_path = layers.Conv2D(64, (1, 1), padding="same", activation="relu")(input_image)

    # Fuse Features
    fused_features = layers.concatenate(main_path + [branch_path])

    # Classification
    x = layers.GlobalAveragePooling2D()(fused_features)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=input_image, outputs=outputs)
    return model