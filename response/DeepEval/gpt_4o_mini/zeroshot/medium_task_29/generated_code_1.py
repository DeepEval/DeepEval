import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for the CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer followed by 1x1 MaxPooling
    x1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x1)
    flatten1 = Flatten()(pool1)

    # Second convolutional layer followed by 2x2 MaxPooling
    x2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x2)
    flatten2 = Flatten()(pool2)

    # Third convolutional layer followed by 4x4 MaxPooling
    x3 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x3)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs from all pooling layers
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(concatenated)
    dense2 = Dense(64, activation='relu')(dense1)

    # Output layer for 10 classes (CIFAR-10)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Display the model architecture