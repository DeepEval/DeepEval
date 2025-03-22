# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Create the main path
    main_path = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2))
    ])

    # Create the branch path
    branch_path = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2))
    ])

    # Combine the main and branch paths through addition
    combined_path = keras.Sequential([
        layers.Add()([main_path.output, branch_path.output]),
        layers.MaxPooling2D((2, 2))
    ])

    # Define the second block with max pooling layers
    max_pooling_path = keras.Sequential([
        layers.MaxPooling2D((1, 1), strides=(1, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.MaxPooling2D((4, 4))
    ])

    # Combine the combined path and max pooling path
    combined_output = keras.layers.Concatenate()([combined_path.output, max_pooling_path(combined_path.output)])

    # Flatten the output
    flattened_output = layers.Flatten()(combined_output)

    # Define the fully connected layers
    fc_path = keras.Sequential([
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Define the final model
    model = keras.Model(inputs=combined_path.input, outputs=fc_path(flattened_output))

    return model

# Create and print the model
model = dl_model()
print(model.summary())