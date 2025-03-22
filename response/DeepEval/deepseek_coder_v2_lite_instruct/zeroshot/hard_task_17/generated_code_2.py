import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Multiply, Reshape

def dl_model():
    # Define the input shape
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    y = Conv2D(64, (3, 3), activation='relu')(x)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D((2, 2))(y)

    # Branch from Block 1
    branch = GlobalAveragePooling2D()(x)
    branch = Dense(32, activation='relu')(branch)
    branch = Dense(32, activation='relu')(branch)
    branch = Reshape((1, 1, 32))(branch)
    branch = Multiply()([x, branch])

    # Fusion of main path and branch
    z = Add()([y, branch])

    # Flatten and classify
    z = GlobalAveragePooling2D()(z)
    outputs = Dense(10, activation='softmax')(z)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model