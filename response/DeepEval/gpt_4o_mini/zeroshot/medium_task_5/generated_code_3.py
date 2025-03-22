import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Main Path
    input_tensor = layers.Input(shape=(32, 32, 3))
    
    # First Block of Convolution in Main Path
    x_main = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x_main = layers.MaxPooling2D(pool_size=(2, 2))(x_main)

    # Second Block of Convolution in Main Path
    x_main = layers.Conv2D(64, (3, 3), activation='relu')(x_main)
    x_main = layers.MaxPooling2D(pool_size=(2, 2))(x_main)

    # Branch Path
    x_branch = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x_branch = layers.MaxPooling2D(pool_size=(2, 2))(x_branch)

    # Combine the Outputs from Main and Branch Paths
    combined = layers.add([x_main, x_branch])

    # Flatten the Combined Output
    x = layers.Flatten()(combined)

    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create the Model
    model = models.Model(inputs=input_tensor, outputs=x)

    return model

# Example of compiling and summarizing the model
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)