import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Define input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Split image into three channel groups
    group1 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    group2 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)
    group3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(inputs)

    # Feature extraction for each group
    conv1a = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(group1)
    conv2a = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1a)
    conv3a = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(conv2a)

    conv1b = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(group2)
    conv2b = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2b)
    conv3b = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(conv2b)

    conv1c = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(group3)
    conv2c = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv2c)
    conv3c = layers.Conv2D(128, (5, 5), padding='same', activation='relu')(conv2c)

    # Concatenate outputs from all groups
    concat = layers.concatenate([conv3a, conv3b, conv3c])

    # Fully connected layers
    fc1 = layers.Dense(512, activation='relu')(concat)
    fc2 = layers.Dense(256, activation='relu')(fc1)
    outputs = layers.Dense(10, activation='softmax')(fc2)

    # Construct model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Compile and train the model (not included in the code)
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)