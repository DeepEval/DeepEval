from keras.models import Model
from keras.layers import Input, Conv2D, Add, Dense


def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)

    main_path = Add()([conv1,conv2,conv3])

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',activation='relu')(input_layer)

    added = Add()([main_path,branch_path])

    dense = Dense(units=128, activation='relu')(added)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

