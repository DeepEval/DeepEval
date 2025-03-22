import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
    adding_layer = Add()([main_path, branch_path])

    conv3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(adding_layer)

    final_output = Add()([conv3, input_layer])

    flatten_layer = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model