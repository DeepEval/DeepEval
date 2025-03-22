from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_tensor = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_tensor = layers.Lambda(lambda x: tf.split(x, 3, axis=2))(input_tensor)

    # Create three parallel branches for feature extraction
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(split_tensor[0])
    branch1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(branch1)

    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(split_tensor[1])
    branch2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(branch2)

    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(split_tensor[2])
    branch3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(branch3)

    # Combine the outputs from the three branches
    combined_features = layers.Add()([branch1, branch2, branch3])

    # Fuse the combined features with the original input
    main_path = layers.Add()([combined_features, input_tensor])

    # Flatten the output and feed it into a fully connected layer
    flatten_layer = layers.Flatten()(main_path)
    output_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_tensor, outputs=output_layer)
    return model