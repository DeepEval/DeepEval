import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate, Lambda, Dropout, Conv2D
from tensorflow import tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # First block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    dropout = Dropout(0.25)(Concatenate()([flat1, flat2, flat3]))

    reshape_layer = tf.keras.layers.Reshape((1, -1))(dropout)

    # Second block
    split_layer = Lambda(lambda x: tf.split(x, 4, axis=1))(reshape_layer)

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    conv4 = Conv2D(filters=256, kernel_size=(7, 7), padding='same', activation='relu')(split_layer[3])

    concat_output = Concatenate(axis=1)([conv1, conv2, conv3, conv4])

    flatten_output = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model