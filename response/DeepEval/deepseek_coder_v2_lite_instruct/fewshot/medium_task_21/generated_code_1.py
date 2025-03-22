import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions (equivalent to a 7x7 with less parameters)
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    flatten = Flatten()(concat)

    # Apply dropout for regularization
    dropout1 = Dropout(0.5)(flatten)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout2)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model