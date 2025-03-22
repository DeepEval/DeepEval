import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Add, Flatten

def dl_model():

    # Input layer for CIFAR-10 dataset
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_mp = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    gap1 = GlobalAveragePooling2D()(conv1_mp)
    fc1 = Dense(units=input_layer.shape[3] * input_layer.shape[1] * input_layer.shape[2], activation='relu')(gap1)
    fc2 = Dense(units=input_layer.shape[3] * input_layer.shape[1] * input_layer.shape[2], activation='relu')(fc1)
    fc2_reshape = Reshape((input_layer.shape[3], input_layer.shape[1], input_layer.shape[2]))(fc2)
    fm_multiply = multiply([input_layer, fc2_reshape])

    # Branch path
    conv_branch = Conv2D(filters=input_layer.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main and branch outputs
    concat = Add()([fm_multiply, conv_branch])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense3 = Dense(units=128, activation='relu')(flatten)
    dense4 = Dense(units=64, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model