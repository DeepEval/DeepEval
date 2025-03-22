from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    # Initial 1x1 Convolution
    main_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)

    # Branch 2: Max Pooling -> 3x3 Convolution -> Upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(main_path)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max Pooling -> 3x3 Convolution -> Upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(main_path)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 Convolution in Main Path
    main_path_output = Conv2D(32, (1, 1), activation='relu')(concatenated_branches)

    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Add the outputs from the main path and the branch path
    combined = Add()([main_path_output, branch_path])

    # Flatten and Fully Connected Layers for Classification
    flat = Flatten()(combined)
    dense1 = Dense(128, activation='relu')(flat)
    outputs = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of instantiating the model
model = dl_model()
model.summary()