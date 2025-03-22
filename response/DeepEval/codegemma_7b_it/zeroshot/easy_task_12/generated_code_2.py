from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    SeparableConv2D,
    MaxPooling2D,
    Conv2D,
    Concatenate,
    Flatten,
    Dense,
)

def dl_model():
    # Input layer
    x = Input(shape=(28, 28, 1))

    # Main path
    main_path = x
    for i in range(2):
        main_path = Activation("relu")(main_path)
        main_path = SeparableConv2D(32, (3, 3), padding="same")(main_path)
        main_path = MaxPooling2D((2, 2), strides=(2, 2))(main_path)

    # Branch path
    branch_path = Conv2D(32, (1, 1), padding="same")(x)

    # Concatenate outputs
    merged = Concatenate()([main_path, branch_path])

    # Flatten and fully connected layer
    merged = Flatten()(merged)
    output = Dense(10, activation="softmax")(merged)

    # Create the model
    model = Model(inputs=x, outputs=output)

    return model