import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1: Convolutional layers and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    # Block 2: Convolutional layers and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Concatenate the outputs of both blocks
    concat = concatenate([pool1, pool2])

    # Flatten the concatenated output
    flat = Flatten()(concat)

    # Fully connected layer
    fc = Dense(units=128, activation='relu')(flat)

    # Output layer
    outputs = Dense(units=10, activation='softmax')(fc)

    # Model construction
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])