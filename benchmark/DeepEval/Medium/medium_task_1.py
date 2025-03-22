from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)

    added = Add()([main_path, input_layer])

    flatten_output = Flatten()(added)
    dense_1 = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense_1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
