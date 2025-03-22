from tensorflow.keras.layers import Input, SeparableConv2D, Conv2D, MaxPooling2D, Flatten, Dense, Add, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def dl_model():
    # Input layer for MNIST images
    inputs = Input(shape=(28, 28, 1))

    # Main path
    # Block 1
    x = SeparableConv2D(32, (3, 3), padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch path
    y = Conv2D(64, (1, 1), padding='same')(inputs)

    # Combine paths
    combined = Add()([x, y])

    # Flatten and Fully Connected Layer
    flat = Flatten()(combined)
    outputs = Dense(10, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Convert class vectors to binary class matrices
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')