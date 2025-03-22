import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    dropout1 = Dropout(0.2)(conv1)

    conv2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(dropout1)
    dropout2 = Dropout(0.2)(conv2)

    conv3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu', padding='same')(dropout2)
    dropout3 = Dropout(0.2)(conv3)

    conv4 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu', padding='same')(dropout3)
    dropout4 = Dropout(0.2)(conv4)

    conv5 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(dropout4) 

    merged = Add()([input_layer, conv5])

    flatten_layer = Flatten()(merged)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model