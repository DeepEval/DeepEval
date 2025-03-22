# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    """
    Creates a deep learning model with multiple branches for enhanced feature extraction
    and three fully connected layers for classification.

    Returns:
        A Keras model instance.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class labels to categorical labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = layers.Dropout(0.2)(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = layers.Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = layers.Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.Dropout(0.2)(branch2)

    # Branch 3: Max pooling
    branch3 = layers.MaxPooling2D((2, 2))(input_layer)

    # Concatenate the outputs from all branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Add fully connected layers
    x = layers.Dense(128, activation='relu')(flattened)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Usage:
model = dl_model()
print(model.summary())