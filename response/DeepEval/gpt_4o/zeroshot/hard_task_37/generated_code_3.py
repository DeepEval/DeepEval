from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input
    input_layer = Input(shape=(28, 28, 1))

    # Function to create a single block
    def create_block(input_tensor):
        # Three sequential convolutional layers
        conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv2)

        # Parallel path with a single convolutional layer directly from input
        parallel_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)

        # Add the outputs of the convolutional layers
        added = Add()([conv1, conv2, conv3, parallel_conv])

        return added

    # Create the two parallel branches
    branch1_output = create_block(input_layer)
    branch2_output = create_block(input_layer)

    # Concatenate the outputs from the two branches
    concatenated = Concatenate()([branch1_output, branch2_output])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layer
    dense = Dense(128, activation='relu')(flattened)

    # Output layer for classification (10 classes for MNIST)
    output = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Example of using the function
model = dl_model()
model.summary()