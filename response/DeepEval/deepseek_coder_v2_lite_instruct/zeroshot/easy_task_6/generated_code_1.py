import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)  # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(x)           # Second convolutional layer
    x = MaxPooling2D((2, 2))(x)                            # Pooling layer

    # Branch path
    branch = Conv2D(32, (1, 1), activation='relu')(input_layer)  # Direct connection to input

    # Addition operation
    x = Add()([x, branch])

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()