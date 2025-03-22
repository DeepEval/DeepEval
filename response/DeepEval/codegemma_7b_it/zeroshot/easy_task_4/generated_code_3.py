import keras
from tensorflow.keras import layers, models

def dl_model():
    # Define the input layer
    input_img = keras.Input(shape=(28, 28, 1))

    # Feature extraction through convolutional and max pooling layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Additional convolutional and max pooling layers
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Classification output layer
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(input_img, output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model