from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Add,
)

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Path 1
    path1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    path1 = AveragePooling2D((2, 2))(path1)
    path1 = Conv2D(64, (3, 3), activation='relu')(path1)
    path1 = AveragePooling2D((2, 2))(path1)

    # Path 2
    path2 = Conv2D(64, (3, 3), activation='relu')(inputs)
    path2 = AveragePooling2D((2, 2))(path2)

    # Combine paths
    outputs = Add()([path1, path2])
    outputs = Flatten()(outputs)

    # Output layer
    outputs = Dense(10, activation='softmax')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model