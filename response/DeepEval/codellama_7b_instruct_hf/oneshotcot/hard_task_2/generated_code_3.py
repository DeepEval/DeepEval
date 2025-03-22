import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(split_layer[0])
    conv_layer2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(split_layer[1])
    conv_layer3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(split_layer[2])
    conv_layer1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_layer1)
    conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_layer2)
    conv_layer3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv_layer3)
    concat_layer = Concatenate()([conv_layer1, conv_layer2, conv_layer3])
    fc_layer = Dense(units=10, activation='softmax')(concat_layer)
    model = keras.Model(inputs=input_layer, outputs=fc_layer)
    return model