import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3)

    # Branch path
    conv4 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_branch = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv4)

    # Combine outputs
    output = Add()([max_pool, max_pool_branch])

    # Classification layer
    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model