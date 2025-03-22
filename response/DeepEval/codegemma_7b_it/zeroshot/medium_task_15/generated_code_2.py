from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, concatenate, multiply, Conv2DTranspose, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv1 = Conv2D(64, (3, 3), padding='same')(input_img)

    # Batch normalization and ReLU activation
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # Global average pooling
    gap = GlobalAveragePooling2D()(act1)

    # Fully connected layers
    fc1 = Dense(64)(gap)
    act2 = Activation('relu')(fc1)
    fc2 = Dense(32)(act2)

    # Reshape and multiplication
    gap = Reshape((1, 1, 64))(gap)
    fc2 = Reshape((1, 1, 32))(fc2)
    weighted_features = multiply([act1, fc2])

    # Concatenation
    concat = concatenate([input_img, weighted_features])

    # 1x1 convolution and average pooling
    conv2 = Conv2D(64, (1, 1), padding='same')(concat)
    act3 = Activation('relu')(conv2)
    gap2 = AveragePooling2D()(act3)

    # Fully connected layer for classification
    fc3 = Dense(10, activation='softmax')(gap2)

    # Model definition
    model = Model(inputs=input_img, outputs=fc3)

    return model