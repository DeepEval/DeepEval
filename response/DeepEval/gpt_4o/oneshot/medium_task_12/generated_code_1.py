import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)

    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)

    # Concatenate with the output of the first block
    concat1 = Concatenate()([relu1, relu2])

    # Third block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(concat1)
    bn3 = BatchNormalization()(conv3)
    relu3 = ReLU()(bn3)

    # Concatenate with the output of the second block
    concat2 = Concatenate()([relu2, relu3])

    # Flatten and dense layers for classification
    flatten_layer = Flatten()(concat2)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model