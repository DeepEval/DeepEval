import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    channel_splits = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    channel_splits = [Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in channel_splits]
    concatenated = Concatenate()(channel_splits)

    # Block 2
    def branch(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(1, 1))(x)
        return x

    branch1 = branch(concatenated)
    branch2 = branch(inputs)
    branch3 = branch(inputs)
    branch3 = MaxPooling2D(pool_size=(1, 1))(branch3)

    # Concatenate all branches
    concatenated_outputs = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Final layers
    x = GlobalAveragePooling2D()(concatenated_outputs)
    output = Dense(units=10, activation='softmax')(x)

    # Model
    model = Model(inputs=inputs, outputs=output)

    return model

model = dl_model()
model.summary()