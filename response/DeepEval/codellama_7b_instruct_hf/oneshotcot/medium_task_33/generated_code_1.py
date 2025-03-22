import keras
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    conv1_layer = SeparableConv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2_layer = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3_layer = SeparableConv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    conv_output = Concatenate()([conv1_layer, conv2_layer, conv3_layer])
    bath_norm = BatchNormalization()(conv_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model