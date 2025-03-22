# Import necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # First sequential block
    block1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    block1 = MaxPooling2D((2, 2))(block1)

    # Second sequential block
    block2 = Conv2D(32, (3, 3), activation='relu')(inputs)
    block2 = MaxPooling2D((2, 2))(block2)

    # Add the outputs of both blocks to the input
    x = Add()([inputs, block1, block2])

    # Flatten the output
    x = Flatten()(x)

    # Define the output layer with a dense layer
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Call the function to get the model
model = dl_model()