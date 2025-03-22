from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First convolutional layer increases the feature map width
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    # Second convolutional layer restores the number of channels
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    # Branch path
    branch = input_layer  # Directly connects to the input

    # Combine both paths using an addition operation
    combined = Add()([x, branch])

    # Flatten the feature maps
    flat = Flatten()(combined)

    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to include channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')