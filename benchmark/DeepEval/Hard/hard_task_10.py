from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Add, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch7x7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch7x7 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch7x7)
    branch7x7 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch7x7)

    concat = Concatenate()([branch1x1, branch7x7])
    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat) 
    added = Add()([input_layer, main_path])

    dense_output = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

