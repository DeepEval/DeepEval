from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer, assuming MNIST images are 28x28 with a single channel (grayscale)
    input_layer = Input(shape=(28, 28, 1))

    # First block: 3 convolutional layers followed by max pooling
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block: 4 convolutional layers followed by max pooling
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Output layer, assuming 10 classes for MNIST
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to get the model and print its summary
model = dl_model()
model.summary()