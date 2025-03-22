import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 Convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1_drop = Dropout(rate=0.25)(conv1_1)

    # Branch 2: 1x1 Convolution + 3x3 Convolution
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2_drop = Dropout(rate=0.25)(conv1_2)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2_drop)
    conv2_2_drop = Dropout(rate=0.25)(conv2_2)

    # Branch 3: 1x1 Convolution + Two 3x3 Convolutions
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3_drop = Dropout(rate=0.25)(conv1_3)
    conv2_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_3_drop)
    conv2_3_drop = Dropout(rate=0.25)(conv2_3)
    conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_3_drop)
    conv3_3_drop = Dropout(rate=0.25)(conv3_3)

    # Branch 4: Average Pooling + 1x1 Convolution
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    conv1_4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    conv1_4_drop = Dropout(rate=0.25)(conv1_4)

    # Concatenate Branch Outputs
    concat_layer = Concatenate()([conv1_1_drop, conv2_2_drop, conv3_3_drop, conv1_4_drop])

    # Fully Connected Layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense1_drop = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dense1_drop)
    dense2_drop = Dropout(rate=0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dense2_drop)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model