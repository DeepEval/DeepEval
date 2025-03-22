from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage with MNIST dataset
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Instantiate and train the model
model = dl_model()
model.summary()
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)