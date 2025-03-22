import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Part 1: Feature extraction
    x = inputs
    for i in range(3):
        x = Conv2D(32 * (2**i), (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    # Part 2: Feature enhancement
    for i in range(2):
        x = Conv2D(64 * (2**i), (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.25)(x)
        x = Conv2D(64 * (2**i), (3, 3), activation='relu', padding='same')(x)

    # Part 3: Upsampling and concatenation
    for i in range(2):
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64 * (2**(3-i)), (3, 3), activation='relu', padding='same')(x)
        x = concatenate([x, inputs])

    # Output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the model
    return model