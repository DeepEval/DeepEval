import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x[1])
    conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x[2])
    conv_output = Concatenate()([conv1, conv2, conv3])
    bn = BatchNormalization()(conv_output)

    # Second block
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(bn)
    path1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    path2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(x)
    path2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    path3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    concat = Concatenate()([path1, path2, path3])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model