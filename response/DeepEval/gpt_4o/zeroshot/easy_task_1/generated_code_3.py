from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer with the shape of MNIST images
    input_layer = Input(shape=(28, 28, 1))

    # First convolutional layer followed by max pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional layer
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    # Flatten the feature maps into a 1D vector
    x = layers.Flatten()(x)

    # First fully connected (dense) layer
    x = layers.Dense(128, activation='relu')(x)

    # Second fully connected (dense) layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer with 10 units for the 10 classes of MNIST
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()

# Load MNIST data for demonstration purposes
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)