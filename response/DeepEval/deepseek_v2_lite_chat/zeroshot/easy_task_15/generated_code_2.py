import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Input

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images to the [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the datasets according to the model's expected input shape
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Convolutional layer followed by max pooling
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 2: Additional convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Apply dropout to the second layer for regularization
    x = Dropout(0.2)(x)
    
    # Block 3: Convolutional layer followed by max pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 4: Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Flatten the output for the fully connected layer
    x = Flatten()(x)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and return the model
model = dl_model()