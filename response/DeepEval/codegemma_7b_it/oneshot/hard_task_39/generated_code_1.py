import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Flatten, Dense, Conv2D, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pooling1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    max_pooling3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    max_pooling_concat = Concatenate()([max_pooling1, max_pooling2, max_pooling3])
    max_pooling_flatten = Flatten()(max_pooling_concat)

    # Block 2
    block_input = Reshape((4, 4, 1))(max_pooling_flatten)

    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_input)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(block_input)
    max_pooling = MaxPooling2D(pool_size=(3, 3), strides=3, padding='same')(block_input)

    block_output = Concatenate()([conv1, conv2, conv3, max_pooling])

    # Classification
    flatten = Flatten()(block_output)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()