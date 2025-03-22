import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First convolutional path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)

    # Second path directly from input
    conv_direct = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Adding outputs from the two paths
    combined = Add()([conv3, conv_direct])

    # Fully connected layers
    flat = Flatten()(combined)
    fc1 = Dense(64, activation='relu')(flat)
    fc2 = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()