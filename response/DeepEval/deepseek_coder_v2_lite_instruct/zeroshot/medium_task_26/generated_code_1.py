import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # 1x1 convolutional layer to compress the input channels
    compressed = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Two parallel convolutional layers: 1x1 and 3x3
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(compressed)
    conv3x3 = Conv2D(32, (3, 3), activation='relu')(compressed)

    # Concatenate the results from the two parallel layers
    concatenated = Concatenate()([conv1x1, conv3x3])

    # Flatten the output feature map into a one-dimensional vector
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()