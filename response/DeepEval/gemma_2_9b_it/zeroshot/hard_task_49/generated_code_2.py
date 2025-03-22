import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    input_img = keras.Input(shape=(28, 28, 1))

    # First Block: Average Pooling
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_img)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    x = tf.keras.layers.Flatten()(x)
    
    x = layers.Concatenate()([x, x, x])  # Concatenate the pooled outputs

    # Fully Connected Layer and Reshape
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((1, 1, 128))(x)

    # Second Block: Depthwise Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(x)
    
    outputs = []
    for i in range(4):
      branch_input = x[i]
      branch_output = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch_input) 
      branch_output = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch_output)
      branch_output = layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(branch_output)
      branch_output = layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same')(branch_output)
      outputs.append(branch_output)

    x = layers.Concatenate(axis=-1)(outputs)
    x = tf.keras.layers.Flatten()(x)
    
    # Output Layer
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_img, outputs=outputs)
    return model