import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)
    conv1_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_3)

    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_2)
    conv2_3 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_3)
    conv2_4 = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_4)

    adding_layer = Concatenate()([main_path, branch_path])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model