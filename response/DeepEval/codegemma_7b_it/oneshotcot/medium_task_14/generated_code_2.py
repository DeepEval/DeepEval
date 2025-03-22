import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1_1 = BatchNormalization()(conv1_1)
    act1_1 = Activation('relu')(bn1_1)

    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    act1_2 = Activation('relu')(bn1_2)

    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1_2)
    bn1_3 = BatchNormalization()(conv1_3)
    act1_3 = Activation('relu')(bn1_3)

    # Block 2
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act1_3)
    bn2_1 = BatchNormalization()(conv2_1)
    act2_1 = Activation('relu')(bn2_1)

    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    act2_2 = Activation('relu')(bn2_2)

    conv2_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2_2)
    bn2_3 = BatchNormalization()(conv2_3)
    act2_3 = Activation('relu')(bn2_3)

    # Block 3
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(act2_3)
    bn3_1 = BatchNormalization()(conv3_1)
    act3_1 = Activation('relu')(bn3_1)

    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(act3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    act3_2 = Activation('relu')(bn3_2)

    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(act3_2)
    bn3_3 = BatchNormalization()(conv3_3)
    act3_3 = Activation('relu')(bn3_3)

    # Parallel branch
    conv_p = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_p = BatchNormalization()(conv_p)
    act_p = Activation('relu')(bn_p)

    # Output paths
    output1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(act3_3)
    output2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(act_p)

    # Concatenation
    concat = Concatenate()([output1, output2])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model