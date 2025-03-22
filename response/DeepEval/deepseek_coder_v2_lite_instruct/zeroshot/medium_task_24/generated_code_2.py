import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(128, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(128, (3, 3), activation='relu')(branch2)

    # Branch 3
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and fully connected layers
    flattened = Flatten()(concatenated)
    fc1 = Dense(128, activation='relu')(flattened)
    dropout1 = Dropout(0.5)(fc1)
    fc2 = Dense(64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(fc2)
    output_layer = Dense(10, activation='softmax')(dropout2)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()