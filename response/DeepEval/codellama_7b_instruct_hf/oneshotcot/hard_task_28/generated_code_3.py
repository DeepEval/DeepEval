import keras
from keras.layers import Input, DepthwiseConv2D, Sequential, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    main_path = Sequential()
    main_path.add(DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=(32, 32, 3)))
    main_path.add(BatchNormalization())
    main_path.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    main_path.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    main_path.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    branch_path = Sequential()
    branch_path.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
    branch_path.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    branch_path.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    output_layer = Concatenate()([main_path.output, branch_path.output])
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=128, activation='relu')(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model