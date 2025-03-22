import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)

    # Process each group sequentially
    group1 = layers.Conv2D(32, 1, activation='relu')(split_tensor[0])
    group1 = layers.Conv2D(64, 3, activation='relu', padding='same')(group1)
    group1 = layers.Conv2D(32, 1, activation='relu')(group1)

    group2 = layers.Conv2D(32, 1, activation='relu')(split_tensor[1])
    group2 = layers.Conv2D(64, 3, activation='relu', padding='same')(group2)
    group2 = layers.Conv2D(32, 1, activation='relu')(group2)

    group3 = layers.Conv2D(32, 1, activation='relu')(split_tensor[2])
    group3 = layers.Conv2D(64, 3, activation='relu', padding='same')(group3)
    group3 = layers.Conv2D(32, 1, activation='relu')(group3)

    # Combine the outputs from the three groups
    combined_features = layers.Add()([group1, group2, group3])

    # Fuse the combined features with the original input
    main_path = layers.Add()([input_tensor, combined_features])

    # Flatten the combined features and feed into a fully connected layer
    output = layers.Flatten()(main_path)
    output = layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_tensor, outputs=output)

    return model