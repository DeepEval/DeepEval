import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First sequential block: 3 convolutional layers followed by max pooling
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Second sequential block: 4 convolutional layers followed by max pooling
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = Conv2D(1024, (3, 3), activation='relu')(x)
    x = Conv2D(2048, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the feature maps
    x = Flatten()(x)

    # Three fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()