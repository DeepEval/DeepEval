import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = Dropout(0.5)(branch1)

    # Branch 2
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.5)(branch2)

    # Branch 3
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch3 = Dropout(0.5)(branch3)

    # Concatenate branches
    combined = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers
    flat = Flatten()(combined)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    outputs = Dense(10, activation='softmax')(dense2)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])