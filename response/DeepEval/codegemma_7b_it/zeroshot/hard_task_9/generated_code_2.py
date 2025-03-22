from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Feature extraction branches
    branch_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch_2)
    branch_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(inputs)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch_3)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch_3)

    # Concatenate branches and adjust output dimensions
    merged = concatenate([branch_1, branch_2, branch_3])
    merged = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(merged)

    # Branch directly connected to input
    shortcut = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(inputs)

    # Fuse main path and branch
    outputs = Add()([merged, shortcut])

    # Classification layers
    outputs = Flatten()(outputs)
    outputs = Dense(256, activation='relu')(outputs)
    outputs = Dense(256, activation='relu')(outputs)
    outputs = Dense(10, activation='softmax')(outputs)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model