import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Block 1: Convolutional Layer and Max Pooling
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: Convolutional Layer and Max Pooling
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and concatenate the outputs from both paths
    x1 = Flatten()(x)
    x2 = Flatten()(x)
    x = concatenate([x1, x2])

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Get the model
model = dl_model()