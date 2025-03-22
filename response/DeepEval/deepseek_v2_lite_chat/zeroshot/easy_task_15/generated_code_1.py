from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data to include channel dimension (for CNN)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (28, 28, 1)

    # First block: Convolution + Pooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x_train[0])
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Second block: Convolution + Pooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Dropout layer
    x = Dropout(0.25)(x)

    # Flatten the tensor output from the pooling layer
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(512, activation='relu')(x)

    # Output layer
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_shape], outputs=[output])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and train the model
model = dl_model()
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)