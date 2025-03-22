import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(bn1)

    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(relu1)
    bn2 = BatchNormalization()(conv2)
    relu2 = Activation('relu')(bn2)

    # Third block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(relu2)
    bn3 = BatchNormalization()(conv3)
    relu3 = Activation('relu')(bn3)

    # Parallel branch processing the input directly
    parallel_conv = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_relu = Activation('relu')(parallel_bn)

    # Combine outputs from all paths
    combined = Add()([relu3, parallel_relu])

    # Fully connected layers for classification
    flatten = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model