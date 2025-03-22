# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the images
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define Path 1
    path1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path1 = layers.BatchNormalization()(path1)

    # Define Path 2
    path2 = layers.AveragePooling2D((2, 2), strides=(2, 2))(inputs)
    path2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(path2)
    path2 = layers.BatchNormalization()(path2)

    # Define Path 3
    path3 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path3 = layers.BatchNormalization()(path3)
    path3_1 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(path3)
    path3_1 = layers.BatchNormalization()(path3_1)
    path3_2 = layers.Conv2D(32, (3, 1), activation='relu', padding='same')(path3)
    path3_2 = layers.BatchNormalization()(path3_2)
    path3 = layers.Concatenate()([path3, path3_1, path3_2])

    # Define Path 4
    path4 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)
    path4 = layers.BatchNormalization()(path4)
    path4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(path4)
    path4 = layers.BatchNormalization()(path4)
    path4_1 = layers.Conv2D(32, (1, 3), activation='relu', padding='same')(path4)
    path4_1 = layers.BatchNormalization()(path4_1)
    path4_2 = layers.Conv2D(32, (3, 1), activation='relu', padding='same')(path4)
    path4_2 = layers.BatchNormalization()(path4_2)
    path4 = layers.Concatenate()([path4, path4_1, path4_2])

    # Concatenate the outputs of the four paths
    outputs = layers.Concatenate()([path1, path2, path3, path4])

    # Define the feature fusion layer
    outputs = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(outputs)
    outputs = layers.BatchNormalization()(outputs)

    # Flatten the outputs
    outputs = layers.Flatten()(outputs)

    # Define the classification layer
    outputs = layers.Dense(64, activation='relu')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model