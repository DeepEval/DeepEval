import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPooling2D, Concatenate, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(conv2)
    pool1 = AvgPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv3)
    pool2 = AvgPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    pool3 = AvgPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv3)
    concat1 = Concatenate()([pool1, pool2, pool3])
    flatten1 = Flatten()(concat1)
    dense1 = Dense(units=128, activation='relu')(flatten1)

    # Block 2
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv6 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(dense1)
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dense1)
    pool4 = AvgPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv8)
    pool5 = AvgPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv8)
    pool6 = AvgPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv8)
    concat2 = Concatenate()([pool4, pool5, pool6])
    flatten2 = Flatten()(concat2)
    dense2 = Dense(units=128, activation='relu')(flatten2)

    # Final output
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model