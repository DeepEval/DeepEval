import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, concatenate, BatchNormalization, Flatten, Dense, add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    split0 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    conv0_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split0[0])
    conv0_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split0[1])
    conv0_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split0[2])
    concat0 = concatenate([conv0_1x1, conv0_3x3, conv0_5x5])

    # Branch path
    conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    combined = add([concat0, conv1_1x1])
    bath_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model