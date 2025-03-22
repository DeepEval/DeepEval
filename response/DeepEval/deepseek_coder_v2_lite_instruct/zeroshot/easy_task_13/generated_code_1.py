import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # First 1x1 convolutional layer
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(inputs)
    x = Dropout(0.25)(x)

    # Second 1x1 convolutional layer
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(0.25)(x)

    # 3x1 convolutional layer
    x = Conv2D(128, kernel_size=(3, 1), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    # 1x3 convolutional layer
    x = Conv2D(256, kernel_size=(1, 3), activation='relu', padding='same')(x)
    x = Dropout(0.25)(x)

    # Restore the number of channels to match the input
    x = Conv2D(1, kernel_size=(1, 1), activation='relu')(x)

    # Add the processed features with the original input
    x = Add()([x, inputs])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    # Output layer
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage:
# model = dl_model()
# model.summary()