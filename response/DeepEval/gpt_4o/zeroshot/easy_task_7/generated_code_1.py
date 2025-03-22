from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

def dl_model():
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels with 1 color channel (grayscale)

    # Main input
    inputs = Input(shape=input_shape)

    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)  # Restore to 1 channel

    # Branch path
    branch = Conv2D(1, (3, 3), activation='relu', padding='same')(inputs)

    # Combine paths
    combined = Add()([x, branch])

    # Output layers
    flat = Flatten()(combined)
    outputs = Dense(10, activation='softmax')(flat)  # 10 classes for MNIST

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
model.summary()