from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28,28,1))

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.5)(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(conv2)
    conv3 = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    main_path = conv3

    added = Add()([main_path, input_layer])

    flatten = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

