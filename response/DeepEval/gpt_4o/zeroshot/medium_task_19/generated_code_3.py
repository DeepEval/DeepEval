from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(64, (1, 1), activation='relu', padding='same')(input_layer)
    branch3 = Conv2D(128, (5, 5), activation='relu', padding='same')(branch3)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(64, (1, 1), activation='relu', padding='same')(branch4)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the output
    flat = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(256, activation='relu')(flat)
    fc2 = Dense(128, activation='relu')(fc1)

    # Output layer for 10 classes
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()

# You can train the model using model.fit() as needed
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)