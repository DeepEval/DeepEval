from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images dimensions

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Convolutional Layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Average Pooling Layers with different window sizes
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten the outputs of the pooling layers
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Concatenate the flattened layers
    concatenated = Concatenate()([flat1, flat2, flat3])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(concatenated)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer for 10 classes
    outputs = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()