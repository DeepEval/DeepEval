import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(32, (1, 1))(input_tensor)
    
    # Branch 1
    branch1 = Conv2D(32, (3, 3))(x)

    # Branch 2
    branch2 = AveragePooling2D((2, 2))(x)
    branch2 = Conv2D(32, (3, 3))(branch2)
    branch2 = Conv2DTranspose(32, (2, 2))(branch2)

    # Branch 3
    branch3 = AveragePooling2D((2, 2))(x)
    branch3 = Conv2D(32, (3, 3))(branch3)
    branch3 = Conv2DTranspose(32, (2, 2))(branch3)

    # Concatenate branches
    x = Concatenate()([branch1, branch2, branch3])
    x = Conv2D(32, (1, 1))(x)

    # Branch Path
    branch_path = Conv2D(32, (1, 1))(input_tensor)

    # Fusion
    x = Add()([x, branch_path])

    # Output Layer
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model