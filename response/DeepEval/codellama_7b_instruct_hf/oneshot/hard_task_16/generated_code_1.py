import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

，增加dl_model 和 return model
def dl_model():
    input_layer = Input(shape=(32, 32, 3))


    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv3)
    output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])


    transition_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)


    branch = Concatenate()([transition_conv, output_tensor])
    branch = BatchNormalization()(branch)
    branch = Flatten()(branch)
    branch = Dense(units=128, activation='relu')(branch)
    branch = Dense(units=64, activation='relu')(branch)
    branch = Dense(units=10, activation='softmax')(branch)


    output_tensor = Add()([branch, output_tensor])


    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)


    model = Model(inputs=input_layer, outputs=output_tensor)

    return model