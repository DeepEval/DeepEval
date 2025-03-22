import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Main path
    # Block 1
    main_path = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    main_path = layers.MaxPooling2D((2, 2))(main_path)

    # Block 2
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.MaxPooling2D((2, 2))(main_path)

    # Branch path
    branch_path = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    branch_path = layers.MaxPooling2D((2, 2))(branch_path)

    # Combine both paths
    combined = layers.add([main_path, branch_path])

    # Flatten the combined output
    flatten = layers.Flatten()(combined)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(dense1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))