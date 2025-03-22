import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 conv followed by 3x3 conv
    branch1_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1_conv1)
    branch1_dropout = Dropout(0.5)(branch1_conv2)

    # Branch 2: 1x1 conv, 1x7 conv, 7x1 conv, 3x3 conv
    branch2_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(branch2_conv1)
    branch2_conv3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(branch2_conv2)
    branch2_conv4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2_conv3)
    branch2_dropout = Dropout(0.5)(branch2_conv4)

    # Branch 3: Max pooling
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch3_dropout = Dropout(0.5)(branch3_pool)

    # Concatenate all branches
    concatenated = Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model