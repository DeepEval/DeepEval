import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Concatenate outputs
    merged = Concatenate(axis=3)([relu1, relu2, relu3])

    # Flatten and fully connected layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model