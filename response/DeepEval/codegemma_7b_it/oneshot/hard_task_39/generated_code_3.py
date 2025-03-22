import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Conv2D, Concatenate, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    max_pooling_flatten = Concatenate()([Flatten()(max_pooling1), Flatten()(max_pooling2), Flatten()(max_pooling3)])

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling_flatten)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling_flatten)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(max_pooling_flatten)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(max_pooling_flatten)

    concat_features = Concatenate()([conv1, conv2, conv3, maxpool])

    reshape_output = Reshape((concat_features.shape[1], concat_features.shape[2], 1))(concat_features)

    # Classification
    flatten = Flatten()(reshape_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model