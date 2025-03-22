import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input layer
    input_layer = keras.layers.Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    x1 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x2 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x3 = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Feature extraction for each group
    x1 = layers.Conv2D(32, (1, 1), activation='relu')(x1)
    x2 = layers.Conv2D(64, (3, 3), activation='relu')(x2)
    x3 = layers.Conv2D(128, (5, 5), activation='relu')(x3)

    # Dropout to reduce overfitting
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dropout(0.2)(x2)
    x3 = layers.Dropout(0.2)(x3)

    # Concatenate the outputs from each group
    x = layers.Concatenate()([x1, x2, x3])

    # Define the second block
    x = layers.Conv2D(128, (1, 1), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(32, (5, 5), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Define the fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model