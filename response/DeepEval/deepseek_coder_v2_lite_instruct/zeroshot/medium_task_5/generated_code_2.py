import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the main path
    main_path = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2))
    ])

    # Define the branch path
    branch_path = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2))
    ])

    # Define the input layer for the model
    inputs = layers.Input(shape=(32, 32, 3))

    # Apply the main path and branch path to the inputs
    main_output = main_path(inputs)
    branch_output = branch_path(inputs)

    # Combine the outputs from both paths using addition
    combined_output = layers.Add()([main_output, branch_output])

    # Flatten the combined output
    flattened_output = layers.Flatten()(combined_output)

    # Add two fully connected layers
    fc1 = layers.Dense(64, activation='relu')(flattened_output)
    fc2 = layers.Dense(10, activation='softmax')(fc1)

    # Construct the final model
    model = models.Model(inputs=inputs, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()