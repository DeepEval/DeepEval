import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras import backend as K

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)

    branch_path = GlobalAveragePooling2D()(main_path)
    flatten1 = Dense(units=128, activation='relu')(branch_path)
    flatten2 = Dense(units=10, activation='relu')(flatten1)
    channel_weights = Multiply()([branch_path, flatten2])
    channel_weights = Reshape(target_shape=(128,))(channel_weights)

    adding_layer = Add()([main_path, channel_weights])
    flatten = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model