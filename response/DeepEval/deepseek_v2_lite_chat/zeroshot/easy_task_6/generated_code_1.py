import keras
from keras.layers import Conv2D, MaxPooling2D, Add, Flatten, Dense, Input

def dl_model():
    # Main path
    input_1 = Input(shape=(28, 28, 1))  # Input layer for the main path

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_1)
    # Branch path
    input_2 = Input(shape=(28, 28, 1))  # Input layer for the branch path

    # Second convolutional layer in the branch path
    conv2 = Conv2D(64, (3, 3), activation='relu')(input_2)

    # Combine paths using an Add layer
    combined = Add()([conv1, conv2])

    # Flatten layer
    flat = Flatten()(combined)

    # Fully connected layer
    dense = Dense(10, activation='softmax')(flat)  # Assuming 10 classes for MNIST

    # Model
    model = keras.Model(inputs=[input_1, input_2], outputs=[dense])

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])