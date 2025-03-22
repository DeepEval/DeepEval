from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    # Define the input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1: Three parallel paths with max pooling layers of different scales
    path1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(path1)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.2)(path1)

    path2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path2)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.2)(path2)

    path3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(path3)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.2)(path3)

    # Concatenate the outputs of the three paths
    concatenated = layers.Concatenate()([path1, path2, path3])

    # Fully connected layer and reshaping operation to transform the output of block 1
    flattened = layers.Reshape((-1,))(concatenated)
    fc1 = layers.Dense(128, activation='relu')(flattened)
    reshaped = layers.Reshape((128, 1))(fc1)

    # Block 2: Four parallel paths with different convolution and pooling strategies
    path4 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    path4 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(path4)
    path4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path4)
    path4 = layers.Flatten()(path4)
    path4 = layers.Dropout(0.2)(path4)

    path5 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    path5 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path5)
    path5 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(path5)
    path5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path5)
    path5 = layers.Flatten()(path5)
    path5 = layers.Dropout(0.2)(path5)

    path6 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    path6 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path6)
    path6 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(path6)
    path6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(path6)
    path6 = layers.Conv2D(32, (7, 1), activation='relu', padding='same')(path6)
    path6 = layers.Conv2D(32, (1, 7), activation='relu', padding='same')(path6)
    path6 = layers.Flatten()(path6)
    path6 = layers.Dropout(0.2)(path6)

    path7 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(reshaped)
    path7 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(path7)
    path7 = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(path7)
    path7 = layers.Flatten()(path7)
    path7 = layers.Dropout(0.2)(path7)

    # Concatenate the outputs of the four paths
    concatenated2 = layers.Concatenate()([path4, path5, path6, path7])

    # Final fully connected layer for classification
    fc2 = layers.Dense(10, activation='softmax')(concatenated2)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=fc2)

    return model