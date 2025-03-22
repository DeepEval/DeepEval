import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    block1 = BatchNormalization()(max_pooling1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1_output = Concatenate()([block1, conv2])
    block2 = BatchNormalization()(block1_output)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2_output = Concatenate()([block2, conv3])
    block3 = BatchNormalization()(block2_output)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3)
    block3_output = Concatenate()([block3, conv4])
    block4 = BatchNormalization()(block3_output)
    flatten_layer = Flatten()(block4)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model