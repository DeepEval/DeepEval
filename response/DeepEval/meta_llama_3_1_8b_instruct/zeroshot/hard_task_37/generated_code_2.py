# Import necessary libraries
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the dl_model function
def dl_model():
    # Create the base model
    inputs = layers.Input(shape=(28, 28, 1))

    # Create a block
    def block(x):
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.AveragePooling2D((2, 2))(x)
        return x

    # Create the main paths
    x1 = block(inputs)
    x2 = block(inputs)

    # Create a parallel branch
    x3 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Combine the paths
    x = layers.Add()([x1, x2, x3])

    # Repeat the block
    x = block(x)

    # Concatenate the outputs
    x = layers.Concatenate()([x1, x2, x])

    # Flatten the output
    x = layers.Flatten()(x)

    # Create the output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = dl_model()

# Print the model summary
print(model.summary())