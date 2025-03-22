import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Add

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1_1)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)

    # Branch path
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Combining both paths
    adding_layer = Add()([pool2_1, pool1_2])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model