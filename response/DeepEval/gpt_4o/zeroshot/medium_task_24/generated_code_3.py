from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    branch1 = Dropout(0.3)(branch1)

    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)
    branch2 = Conv2D(32, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = Dropout(0.3)(branch2)

    # Branch 3: <max pooling>
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch3 = Dropout(0.3)(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and add fully connected layers
    flat = Flatten()(concatenated)
    fc1 = Dense(256, activation='relu')(flat)
    fc2 = Dense(128, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=fc3)

    return model

# Example usage:
# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))