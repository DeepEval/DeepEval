import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # Second Block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    # Third Block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    # Parallel Branch
    parallel_conv = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    parallel_bn = BatchNormalization()(parallel_conv)
    parallel_act = Activation('relu')(parallel_bn)

    # Adding outputs from all paths
    add_layer = Add()([act1, act2, act3, parallel_act])

    # Classification part
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model