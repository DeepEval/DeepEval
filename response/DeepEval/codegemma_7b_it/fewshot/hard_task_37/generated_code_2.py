import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_2)

    # Branch 2
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv2_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_2)

    # Branch 3
    conv3_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3_1)
    conv3_2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(pool3)
    conv3_3 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(conv3_2)

    # Branch 4
    conv4 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add outputs of all branches
    add_output = Add()([conv1_3, conv2_3, conv3_3, conv4])

    # Flatten and fully connected layer
    flatten = Flatten()(add_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model