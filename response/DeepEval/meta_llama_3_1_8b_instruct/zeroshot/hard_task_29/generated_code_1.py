# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the main path of the first block
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(main_path)

    # Define the branch path of the first block
    branch_path = inputs

    # Combine the outputs from both paths
    combined_path = layers.Add()([main_path, branch_path])

    # Apply three max pooling layers with varying scales
    pooled_1 = layers.MaxPooling2D((1, 1), strides=1)(combined_path)
    pooled_2 = layers.MaxPooling2D((2, 2), strides=2)(pooled_1)
    pooled_3 = layers.MaxPooling2D((4, 4), strides=4)(pooled_2)

    # Flatten the results from the pooling layers
    flat_1 = layers.Flatten()(pooled_1)
    flat_2 = layers.Flatten()(pooled_2)
    flat_3 = layers.Flatten()(pooled_3)

    # Concatenate the flattened vectors
    concatenated = layers.Concatenate()([flat_1, flat_2, flat_3])

    # Define the fully connected layers
    fc_1 = layers.Dense(128, activation='relu')(concatenated)
    outputs = layers.Dense(10, activation='softmax')(fc_1)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
# Print a summary of the model
print(model.summary())