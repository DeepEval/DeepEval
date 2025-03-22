# Import necessary packages
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import Conv2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def dl_model():
    # Define the input shape of the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the main pathway
    main_pathway_input = Input(shape=input_shape)

    # Extract spatial features using a 3x3 convolutional layer
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(main_pathway_input)

    # Integrate inter-channel information using two 1x1 convolutional layers
    x = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(x)

    # Apply max pooling to reduce the size of the feature maps
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Mitigate overfitting using a 50% dropout layer
    x = layers.Dropout(0.5)(x)

    # Define the branch pathway
    branch_pathway_input = Input(shape=input_shape)
    y = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(branch_pathway_input)
    y = layers.MaxPooling2D(pool_size=(2, 2))(y)
    y = layers.Dropout(0.5)(y)

    # Fuse the outputs from both pathways using an addition operation
    x = layers.Add()([x, y])

    # Apply global average pooling to reduce the spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Apply a fully connected layer for classification
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = models.Model(inputs=[main_pathway_input, branch_pathway_input], outputs=x)

    return model

# Compile the model
def compile_model(model):
    # Define the loss function and optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

# Load the MNIST dataset
def load_mnist_dataset():
    # Load the training and testing datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape the input data
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, y_train, x_test, y_test

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    # Train the model for 10 epochs
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

# Main function
def main():
    # Construct the deep learning model
    model = dl_model()

    # Compile the model
    compiled_model = compile_model(model)

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test = load_mnist_dataset()

    # Train the model
    train_model(compiled_model, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()