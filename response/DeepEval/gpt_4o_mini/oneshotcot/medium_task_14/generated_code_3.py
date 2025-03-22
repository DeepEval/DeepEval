import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(batch_norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    batch_norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(batch_norm2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(relu2)
    batch_norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(batch_norm3)

    # Direct convolutional branch processing the input
    direct_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm_direct = BatchNormalization()(direct_conv)
    relu_direct = ReLU()(batch_norm_direct)

    # Combining the outputs from the three blocks and the direct branch
    combined = Add()([relu1, relu2, relu3, relu_direct])

    # Flatten and dense layers for classification
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # Output layer for 10 classes in CIFAR-10

    model = Model(inputs=input_layer, outputs=dense2)

    return model