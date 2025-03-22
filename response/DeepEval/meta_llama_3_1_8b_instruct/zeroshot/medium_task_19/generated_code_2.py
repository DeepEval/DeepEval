# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer with shape (32, 32, 3) representing CIFAR-10 images
    input_layer = keras.Input(shape=(32, 32, 3))

    # First branch: 1x1 convolution for dimensionality reduction
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Second branch: extract features by applying 1x1 convolution followed by 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)

    # Third branch: capture larger spatial information by applying 1x1 convolution followed by 5x5 convolution
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch3 = layers.Conv2D(32, (5, 5), activation='relu')(branch3)

    # Fourth branch: perform 3x3 max pooling for downsampling, followed by 1x1 convolution
    branch4 = layers.MaxPooling2D((3, 3))(input_layer)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs of the branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the features
    flattened = layers.Flatten()(concatenated)

    # Add fully connected layers for classification
    x = layers.Dense(64, activation='relu')(flattened)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model

# Load the CIFAR-10 dataset
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Compile and train the model
model = dl_model()
model = compile_model(model)
model = train_model(model, X_train, y_train, X_test, y_test)

# Save the model
model.save('cifar10_model.h5')