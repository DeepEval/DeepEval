# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the Functional API model
    inputs = keras.Input(shape=input_shape)

    # Define the first block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Define the second block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Define the third block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Concatenate the output of each block along the channel dimension
    x = layers.Concatenate()([layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs),
                             x])

    x = layers.Concatenate()([layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x),
                             x])

    # Flatten the output
    x = layers.Flatten()(x)

    # Define the fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model

# Compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])