import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)
    x1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x2 = Conv2D(32, (5, 5), activation='relu')(input_layer)
    x = Concatenate()([x, x1, x2])
    x = Dropout(0.5)(x)

    # Block 2
    branch1 = Conv2D(32, (1, 1), activation='relu')(x)
    branch2 = Conv2D(32, (1, 1), activation='relu')(x)
    branch3 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch4 = Conv2D(32, (5, 5), activation='relu')(branch2)
    branch5 = Conv2D(32, (3, 3), activation='relu')(MaxPooling2D((3, 3))(x))
    branch6 = Conv2D(32, (1, 1), activation='relu')(branch5)
    x = Concatenate()([branch1, branch3, branch4, branch6])

    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()