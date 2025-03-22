# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    The model features two pathways for processing features.

    Returns:
        model: The constructed deep learning model.
    """

    # Define the input shape of the model
    input_shape = (28, 28, 1)

    # Define the main pathway
    main_pathway = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5)
    ])

    # Define the branch pathway
    branch_pathway = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2))
    ])

    # Fuse the outputs from both pathways
    fused_output = layers.Add()([main_pathway.output, branch_pathway.output])

    # Define the output pathway
    output_pathway = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Define the model
    model = Model(inputs=main_pathway.input, outputs=output_pathway(fused_output))

    return model