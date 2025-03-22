# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the images in the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = layers.Input(shape=input_shape)
    block1 = layers.SeparableConv2D(32, (3, 3), activation='relu')(main_path)
    block1 = layers.MaxPooling2D((2, 2))(block1)
    block2 = layers.SeparableConv2D(64, (3, 3), activation='relu')(block1)
    block2 = layers.MaxPooling2D((2, 2))(block2)

    # Define the branch path
    branch_path = layers.Input(shape=input_shape)
    branch_path = layers.Conv2D(64, (1, 1), activation='relu')(branch_path)

    # Sum the outputs from both paths
    merged = layers.Add()([block2, branch_path])

    # Apply a flattening layer and a fully connected layer to generate the final probability distribution
    output = layers.Flatten()(merged)
    output = layers.Dense(10, activation='softmax')(output)

    # Define the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=output)

    return model