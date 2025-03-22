from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    num_classes = 10  # CIFAR-10 has 10 classes

    inputs = Input(shape=input_shape)

    # Path 1: Single 1x1 Convolution
    path1 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)

    # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
    path2 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
    path2 = Conv2D(64, (1, 7), activation='relu', padding='same')(path2)
    path2 = Conv2D(64, (7, 1), activation='relu', padding='same')(path2)

    # Path 3: 1x1 Convolution followed by two sets of 1x7 and 7x1 Convolutions
    path3 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
    path3 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3)
    path3 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3)
    path3 = Conv2D(64, (1, 7), activation='relu', padding='same')(path3)
    path3 = Conv2D(64, (7, 1), activation='relu', padding='same')(path3)

    # Path 4: Average Pooling followed by a 1x1 Convolution
    path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    path4 = Conv2D(64, (1, 1), activation='relu', padding='same')(path4)

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the output and pass it through a fully connected layer for classification
    flat = Flatten()(concatenated)
    outputs = Dense(num_classes, activation='softmax')(flat)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model