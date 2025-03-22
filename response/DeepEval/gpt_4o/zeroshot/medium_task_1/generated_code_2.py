from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

    # Max pooling layer
    pool = MaxPooling2D((2, 2))(conv2)

    # Add the output of the max pooling layer with the input layer
    # To match dimensions, we apply a convolution to the input layer
    input_adjusted = Conv2D(64, (1, 1), padding='same')(input_layer)
    added = Add()([pool, input_adjusted])

    # Flatten the features
    flat = Flatten()(added)

    # First fully connected (dense) layer
    dense1 = Dense(128, activation='relu')(flat)

    # Second fully connected (dense) layer
    dense2 = Dense(64, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Loading and preparing the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Getting the model
model = dl_model()
# Summary of the model architecture
model.summary()

# Training the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))