from keras.models import Model
from keras.layers import Input, Activation, SeparableConv2D, MaxPooling2D, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    activated_1 = Activation('relu')(input_layer)
    separableconv_1 = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same')(activated_1)
    activated_2 = Activation('relu')(separableconv_1)
    separableconv_2 = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding='same')(activated_2)

    main_path = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(separableconv_2)
    branch_path = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2), padding='same')(input_layer)
    added = Add()([branch_path,main_path])

    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
