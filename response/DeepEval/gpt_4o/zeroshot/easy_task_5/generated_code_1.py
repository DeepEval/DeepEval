import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape
    inputs = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with 1 color channel

    # Step 1: Reduce the dimensionality using a 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)

    # Step 2: Extract features using a 3x3 convolution
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

    # Step 3: Restore the dimensionality with another 1x1 convolution
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer with 10 neurons for classification
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
# Display the model's architecture
model.summary()