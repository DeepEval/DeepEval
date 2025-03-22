import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the input shape
input_shape = (32, 32, 3)

# Define the main path
main_path = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Define the branch path
branch_path = keras.Sequential([
    layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Define the combined model
model = keras.models.Model(inputs=main_path.inputs, outputs=main_path.outputs + branch_path.outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

return model