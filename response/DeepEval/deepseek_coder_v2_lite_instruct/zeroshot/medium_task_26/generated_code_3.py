import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    inputs = Input(shape=(32, 32, 64))

    # 1x1 convolutional layer to compress the input channels
    compressed = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Two parallel convolutional layers
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(compressed)
    conv3x3 = Conv2D(32, (3, 3), activation='relu')(compressed)

    # Concatenate the results of the two parallel layers
    concatenated = concatenate([conv1x1, conv3x3])

    # Flatten the output feature map
    flattened = Flatten()(concatenated)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)  # Assuming 10 classes for classification

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    return model

# Example usage:
# model = dl_model()
# model.summary()