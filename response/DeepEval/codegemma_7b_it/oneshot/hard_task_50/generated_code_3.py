import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Lambda, SeparableConv2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First block
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    flatten1 = Flatten()(max_pooling1)
    flatten2 = Flatten()(max_pooling2)
    flatten3 = Flatten()(max_pooling3)

    concat = Concatenate()([flatten1, flatten2, flatten3])
    drop = Dropout(rate=0.2)(concat)

    reshape = keras.layers.Reshape((1, 1, drop.shape[1]))(drop)

    # Second block
    split = Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)

    conv1 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x))(split[0])
    conv2 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x))(split[1])
    conv3 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(x))(split[2])
    conv4 = Lambda(lambda x: SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(x))(split[3])

    concat_conv = Concatenate()([conv1, conv2, conv3, conv4])

    flatten_conv = Flatten()(concat_conv)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_conv)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model