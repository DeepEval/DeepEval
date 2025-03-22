import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the 1x1 convolution branch
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the 1x1 convolution followed by 3x3 convolution branch
    branch2 = layers.Conv2D(64, (1, 1), activation='relu')(input_shape)
    branch2 = layers.Conv2D(64, (3, 3), activation='relu')(branch2)

    # Define the 1x1 convolution followed by two consecutive 3x3 convolutions branch
    branch3 = layers.Conv2D(64, (1, 1), activation='relu')(input_shape)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), activation='relu')(branch3)

    # Define the average pooling followed by 1x1 convolution branch
    branch4 = layers.AveragePooling2D((2, 2))(input_shape)
    branch4 = layers.Conv2D(64, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from all branches
    concatenated = layers.concatenate([branch1, branch2, branch3, branch4])

    # Add dropout layers to mitigate overfitting
    concatenated = layers.Dropout(0.25)(concatenated)
    concatenated = layers.Dropout(0.25)(concatenated)

    # Add three fully connected layers for classification
    outputs = layers.Dense(10, activation='softmax')(concatenated)
    outputs = layers.Dense(10, activation='softmax')(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Create the model
    model = models.Model(inputs=input_shape, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model