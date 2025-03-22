from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    input_channels = input_layer.shape[-1]
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid',activation='relu')(input_layer)
    conv1 = Dropout(rate=0.5)(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu')(conv1)
    conv2 = Dropout(rate=0.5)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    conv3 = Dropout(rate=0.5)(conv3)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same',activation='relu')(conv3)
    conv4 = Dropout(rate=0.5)(conv4)
    main_path = Conv2D(filters=input_channels, kernel_size=(1, 1), strides=(1, 1), padding='same',activation='relu')(conv4)

    added = Add()([main_path, input_layer])

    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
