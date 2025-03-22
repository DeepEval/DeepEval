from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Add, MaxPooling2D, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1: Sequential convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    conv1 = MaxPooling2D()(x)

    # Path 2: Separate convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(inputs)
    conv2 = MaxPooling2D()(x)

    # Concatenate outputs from both paths
    concat = Add()([conv1, conv2])

    # Fully connected layers for classification
    x = Flatten()(concat)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model
