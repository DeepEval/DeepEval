from tensorflow.keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    concat1 = Concatenate()([input_layer, conv1])

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    concat2 = Concatenate()([concat1, conv2])

    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)
    concat3 = Concatenate()([concat2, conv3])

    # Flatten the output from the last concatenation
    flat = Flatten()(concat3)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Get the model
model = dl_model()

# Print the model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))