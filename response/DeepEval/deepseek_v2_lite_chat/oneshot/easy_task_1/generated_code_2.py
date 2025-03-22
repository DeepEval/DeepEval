import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # First Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Second Convolutional Layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Flatten layer
    flatten = Flatten()(pool2)

    # Fully connected layer 1
    fc1 = Dense(units=128, activation='relu')(flatten)

    # Fully connected layer 2
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Final Output Layer
    output = Dense(units=10, activation='softmax')(fc2)

    # Construct the model
    model = Model(inputs=inputs, outputs=output)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])