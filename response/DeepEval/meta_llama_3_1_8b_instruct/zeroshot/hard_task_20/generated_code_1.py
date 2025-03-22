# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    def main_path(x):
        # Split the input into three groups
        groups = tf.split(x, 3, axis=1)
        
        # Apply feature extraction with convolutional layers of different kernel sizes
        group1 = layers.Conv2D(32, (1, 1), activation='relu')(groups[0])
        group2 = layers.Conv2D(32, (3, 3), activation='relu')(groups[1])
        group3 = layers.Conv2D(32, (5, 5), activation='relu')(groups[2])
        
        # Concatenate the outputs from the three groups
        concatenated = layers.Concatenate()([group1, group2, group3])
        
        return concatenated

    # Define the branch path
    def branch_path(x):
        # Apply a 1x1 convolutional layer to align the number of output channels
        x = layers.Conv2D(32, (1, 1), activation='relu')(x)
        return x

    # Create the model
    inputs = keras.Input(shape=input_shape)
    x = main_path(inputs)
    x = branch_path(x)
    x = layers.Add()([x, inputs])  # Combine the outputs of the main and branch paths
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()