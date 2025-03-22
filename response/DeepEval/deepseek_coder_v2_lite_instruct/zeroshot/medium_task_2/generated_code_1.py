import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Define the main pathway
    input_main = Input(shape=(32, 32, 3))
    x_main = Conv2D(32, (3, 3), activation='relu')(input_main)
    x_main = Conv2D(32, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)

    # Define the branch pathway
    input_branch = Input(shape=(32, 32, 3))
    x_branch = Conv2D(32, (5, 5), activation='relu')(input_branch)

    # Combine the outputs from both pathways
    combined = concatenate([x_main, x_branch])

    # Flatten the combined features
    x = Flatten()(combined)

    # Add two fully connected layers
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    return model

# Example usage:
# model = dl_model()
# model.summary()