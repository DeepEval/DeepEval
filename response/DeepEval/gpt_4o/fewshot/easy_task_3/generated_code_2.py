import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First <conv, conv, maxpool> block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Second <conv, conv, maxpool> block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Third <conv, conv, conv, maxpool> block
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_3)

    # Fourth <conv, conv, conv, maxpool> block
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_3)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(pool4)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model