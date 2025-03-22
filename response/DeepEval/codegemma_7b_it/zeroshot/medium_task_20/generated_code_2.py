from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)

    # Path 2: 3x3 convolutions
    path2 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    path2 = Conv2D(64, (3, 3), padding='same', activation='relu')(path2)

    # Path 3: 3x3 convolution
    path3 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_img)
    path3 = Conv2D(64, (3, 3), padding='same', activation='relu')(path3)

    # Path 4: Max pooling followed by 1x1 convolution
    path4 = MaxPooling2D((2, 2))(input_img)
    path4 = Conv2D(64, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate paths
    concat = Concatenate()([path1, path2, path3, path4])

    # Flatten and dense layer
    flatten = Flatten()(concat)
    dense = Dense(128, activation='relu')(flatten)

    # Output layer
    output = Dense(10, activation='softmax')(dense)

    # Create model
    model = Model(inputs=input_img, outputs=output)

    return model