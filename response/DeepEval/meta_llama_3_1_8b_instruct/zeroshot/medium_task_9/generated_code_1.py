from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the initial convolutional layer
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(layers.Input(shape=input_shape))

    # Define the basic block
    def basic_block(x):
        residual = x
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        return x

    # Define the main structure of the model
    x = basic_block(x)
    x = basic_block(x)

    # Define the branch
    branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Feature fusion
    x = layers.Add()([x, branch])

    # Average pooling layer
    x = layers.AveragePooling2D((2, 2))(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Define the fully connected layer for classification
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=x, outputs=x)

    return model

# Create an instance of the model
model = dl_model()
model.summary()