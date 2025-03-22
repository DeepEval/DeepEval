from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout2D,
    Add,
    GlobalAveragePooling2D,
)

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Dropout2D(0.5)(x)

    # Branch pathway
    branch = Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Fusion
    fused = Add()([x, branch])

    # Classification
    x = GlobalAveragePooling2D()(fused)
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output)

    return model