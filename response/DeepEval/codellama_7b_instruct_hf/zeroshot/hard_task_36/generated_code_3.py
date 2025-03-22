import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first pathway
    first_pathway = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.Conv2D(32, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.Conv2D(64, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
    ])

    # Define the second pathway
    second_pathway = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(16, (1, 1), activation='relu'),
        layers.Conv2D(16, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
    ])

    # Define the fusion layer
    fusion_layer = layers.Add()([first_pathway, second_pathway])

    # Define the output layer
    output_layer = layers.GlobalAveragePooling2D()(fusion_layer)
    output_layer = layers.Flatten()(output_layer)
    output_layer = layers.Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = keras.Model(inputs=first_pathway.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model