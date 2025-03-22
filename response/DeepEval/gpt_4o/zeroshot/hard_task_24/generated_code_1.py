from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1: 3x3 convolutional layer
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Branch 2: MaxPooling, 3x3 convolutional, then UpSampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: MaxPooling, 3x3 convolutional, then UpSampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(x)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # 1x1 convolutional layer after concatenation
    x = Conv2D(128, (1, 1), activation='relu')(concatenated)

    # Flattening and fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model