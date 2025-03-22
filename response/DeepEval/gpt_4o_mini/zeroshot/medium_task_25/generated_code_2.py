import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = Input(shape=input_shape)

    # Path 1: 1x1 Convolution
    path1 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Path 2: Average Pooling + 1x1 Convolution
    path2 = AveragePooling2D(pool_size=(2, 2), padding='same')(inputs)
    path2 = Conv2D(32, (1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 Convolution + (1x3 Convolution || 3x1 Convolution)
    path3 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path3_1 = Conv2D(32, (1, 3), padding='same', activation='relu')(path3)
    path3_2 = Conv2D(32, (3, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3_1, path3_2])

    # Path 4: 1x1 Convolution + 3x3 Convolution + (1x3 Convolution || 3x1 Convolution)
    path4 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path4 = Conv2D(32, (3, 3), padding='same', activation='relu')(path4)
    path4_1 = Conv2D(32, (1, 3), padding='same', activation='relu')(path4)
    path4_2 = Conv2D(32, (3, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4_1, path4_2])

    # Concatenate all paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Fully connected layer for classification
    x = Flatten()(concatenated)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example of using the model
model = dl_model()
model.summary()  # To visualize the model architecture