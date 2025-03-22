import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv1_1 = Conv2D(filters=32, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=1, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    main_path = Add()([conv1_2, input_layer])

    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Add()([conv2_1, main_path])

    flatten_layer = Flatten()(branch_path)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model