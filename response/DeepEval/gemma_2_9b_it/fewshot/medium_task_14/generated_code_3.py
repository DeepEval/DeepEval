import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1_1 = BatchNormalization()(conv1_1)
    act1_1 = Activation('relu')(bn1_1)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(act1_1)

    # Block 2
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool1_1)
    bn2_1 = BatchNormalization()(conv2_1)
    act2_1 = Activation('relu')(bn2_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(act2_1)

    # Block 3
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(pool2_1)
    bn3_1 = BatchNormalization()(conv3_1)
    act3_1 = Activation('relu')(bn3_1)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(act3_1)

    # Parallel Branch
    conv_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_branch = BatchNormalization()(conv_branch)
    act_branch = Activation('relu')(bn_branch)

    # Merge paths
    merged_output = Add()([pool3_1, act_branch])

    # Flatten and Classify
    flatten = Flatten()(merged_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model