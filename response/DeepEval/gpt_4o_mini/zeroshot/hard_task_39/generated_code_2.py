import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = layers.Input(shape=(28, 28, 1))

    # Block 1
    # Max Pooling layers with varying scales
    max_pool_1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    max_pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    max_pool_3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs of the pooling layers
    flat_1 = layers.Flatten()(max_pool_1)
    flat_2 = layers.Flatten()(max_pool_2)
    flat_3 = layers.Flatten()(max_pool_3)

    # Concatenate the flattened outputs
    concat_block1 = layers.concatenate([flat_1, flat_2, flat_3])

    # Fully connected layer before Block 2
    dense_layer = layers.Dense(128, activation='relu')(concat_block1)
    
    # Reshape to 4D tensor for Block 2
    reshaped_layer = layers.Reshape((1, 1, 128))(dense_layer)

    # Block 2
    # Branches for feature extraction
    conv_1x1 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped_layer)
    conv_3x3 = layers.Conv2D(32, (3, 3), activation='relu')(reshaped_layer)
    conv_5x5 = layers.Conv2D(32, (5, 5), activation='relu')(reshaped_layer)
    max_pool_3x3 = layers.MaxPooling2D(pool_size=(3, 3))(reshaped_layer)

    # Concatenate the outputs of all branches
    concat_block2 = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool_3x3])

    # Flatten the concatenated output
    flatten_output = layers.Flatten()(concat_block2)

    # Fully connected layer for classification
    output_layer = layers.Dense(10, activation='softmax')(flatten_output)

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
if __name__ == "__main__":
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Create the model
    model = dl_model()

    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')