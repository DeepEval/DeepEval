import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


input_shape = (32, 32, 3)


# Main path
main_path = keras.Sequential([
    layers.Lambda(lambda x: tf.split(x, 3, axis=3)),
    layers.Conv2D(32, (1, 1), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu')
])

# Branch path
branch_path = keras.Sequential([
    layers.Conv2D(32, (1, 1), padding='same', activation='relu')
])


# Concatenate main and branch paths
merged_path = layers.Concatenate()([main_path, branch_path])


# Batch normalization
batch_norm = layers.BatchNormalization()(merged_path)

# Flatten output
flat = layers.Flatten()(batch_norm)


# Fully connected layers
fc1 = layers.Dense(128, activation='relu')(flat)
fc2 = layers.Dense(10, activation='softmax')(fc1)


model = keras.Model(inputs=input_shape, outputs=fc2)
return model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define input shape
input_shape = (32, 32, 3)

# Define main path
main_path = keras.Sequential([
    layers.Lambda(lambda x: tf.split(x, 3, axis=3)),
    layers.Conv2D(32, (1, 1), padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu')
])

# Define branch path
branch_path = keras.Sequential([
    layers.Conv2D(32, (1, 1), padding='same', activation='relu')
])

# Concatenate main and branch paths
merged_path = layers.Concatenate()([main_path, branch_path])

# Batch normalization
batch_norm = layers.BatchNormalization()(merged_path)

# Flatten output
flat = layers.Flatten()(batch_norm)

# Fully connected layers
fc1 = layers.Dense(128, activation='relu')(flat)
fc2 = layers.Dense(10, activation='softmax')(fc1)

# Define model
model = keras.Model(inputs=input_shape, outputs=fc2)

# Return model
return model