import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    norm1 = BatchNormalization()(conv1)
    relu1 = ReLU()(norm1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)
    relu2 = ReLU()(norm2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(relu2)
    norm3 = BatchNormalization()(conv3)
    relu3 = ReLU()(norm3)

    # Parallel branch
    parallel_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    parallel_norm = BatchNormalization()(parallel_conv)
    parallel_relu = ReLU()(parallel_norm)

    # Adding outputs of all paths
    added_output = Add()([relu1, relu2, relu3, parallel_relu])

    # Flatten and Fully Connected Layers
    flatten = Flatten()(added_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=dense2)

    return model