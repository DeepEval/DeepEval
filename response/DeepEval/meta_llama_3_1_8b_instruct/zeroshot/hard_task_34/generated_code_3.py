# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations

def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = keras.Sequential()
    main_path.add(layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    for _ in range(2):
        main_path.add(layers.SeparableConv2D(32, (3, 3), activation='relu'))
        main_path.add(layers.MaxPooling2D((2, 2)))

    # Define the branch path
    branch_path = keras.Sequential()
    branch_path.add(layers.Conv2D(32, (3, 3), activation='relu'))

    # Define the model by concatenating the main path and the branch path, and then adding a fully connected layer
    model = keras.Sequential([
        layers.Concatenate()([main_path, branch_path]),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model