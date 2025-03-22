import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    dropout1 = Dropout(0.25)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(dropout1)
    dropout2 = Dropout(0.25)(conv2)

    conv3 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(dropout2)
    conv4 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(conv3)

    upconv = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv4)

    output = Add()([input_layer, upconv])

    flatten = Flatten()(output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model