import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    main_output = max_pooling

    branch_output = GlobalAveragePooling2D()(max_pooling)
    branch_output = Dense(units=64, activation='relu')(branch_output)
    branch_output = Dense(units=10, activation='relu')(branch_output)
    branch_output = Reshape((10,))(branch_output)
    branch_output = Multiply()([branch_output, main_output])

    output = Concatenate()([main_output, branch_output])
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model