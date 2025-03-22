import keras
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch1_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    branch2_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_output = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_output)

    branch3_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_output = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3_output)

    branch4_output = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(input_layer)
    branch4_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4_output)

    output_tensor = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])
    output_tensor = Flatten()(output_tensor)

    output_layer = Dense(units=64, activation='relu')(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model