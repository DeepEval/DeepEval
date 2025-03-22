import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate

def dl_model():

    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
    path2 = Conv2D(16, (1, 7), padding='same', activation='relu')(path2)
    path2 = Conv2D(16, (7, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
    path3 = Conv2D(16, (1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(16, (7, 1), padding='same', activation='relu')(path3)
    path3 = Conv2D(16, (1, 7), padding='same', activation='relu')(path3)
    path3 = Conv2D(16, (7, 1), padding='same', activation='relu')(path3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4 = AveragePooling2D((2, 2), padding='same')(input_img)
    path4 = Conv2D(16, (1, 1), padding='same', activation='relu')(path4)

    # Concatenate outputs from all paths
    concat = concatenate([path1, path2, path3, path4])

    # Fully connected layer for classification
    x = Flatten()(concat)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_img, outputs=x)

    return model