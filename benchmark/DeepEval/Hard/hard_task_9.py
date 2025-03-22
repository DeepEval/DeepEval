from keras.models import Model
from keras.layers import Input, Conv2D, Concatenate, Add, Dense


def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    concat = Concatenate()([branch1, branch2, branch3])

    main_path = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same')(concat) 
    added = Add()([main_path,input_layer])
    
    dense_1 = Dense(units=128, activation='relu')(added)
    dense_2 = Dense(units=128, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
