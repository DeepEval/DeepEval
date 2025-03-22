import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Reshape, Dense, Concatenate

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    drop1 = Dropout(0.2)(flat1)
    drop2 = Dropout(0.2)(flat2)
    drop3 = Dropout(0.2)(flat3)

    concat1 = Concatenate()([drop1, drop2, drop3])

    # Fully connected and reshape
    dense1 = Dense(128, activation='relu')(concat1)
    reshape_layer = Reshape((4, 32))(dense1)

    # Block 2
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)

    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    pool_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshape_layer)
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool_1)

    concat2 = Concatenate()([conv1_2, conv2_3, conv3_1])

    # Final classification layers
    flatten_output = Flatten()(concat2)
    dense2 = Dense(64, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model