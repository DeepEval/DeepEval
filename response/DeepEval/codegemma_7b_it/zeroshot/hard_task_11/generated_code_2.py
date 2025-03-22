from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, concatenate, add

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Main pathway
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Parallel branch
    parallel_branch = Conv2D(32, (1, 1), activation='relu', padding='same')(input_img)
    parallel_branch = Conv2D(32, (1, 3), activation='relu', padding='same')(parallel_branch)
    parallel_branch = Conv2D(32, (3, 1), activation='relu', padding='same')(parallel_branch)

    # Concatenate outputs
    concat = concatenate([x, parallel_branch])

    # Final 1x1 convolution
    concat = Conv2D(128, (1, 1), activation='relu', padding='same')(concat)

    # Direct connection from input
    shortcut = Conv2D(128, (1, 1), activation='relu', padding='same')(input_img)

    # Additive fusion
    output = add([concat, shortcut])

    # Fully connected layers
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Create model
    model = Model(inputs=input_img, outputs=output)

    return model