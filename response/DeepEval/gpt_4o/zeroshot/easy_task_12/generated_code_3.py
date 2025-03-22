import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Main Path: Block 1
    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Main Path: Block 2
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch Path
    branch = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)

    # Add outputs from main and branch paths
    x = Add()([x, branch])

    # Flatten the output and add a fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating and compiling the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# You can now train the model using model.fit(), etc.