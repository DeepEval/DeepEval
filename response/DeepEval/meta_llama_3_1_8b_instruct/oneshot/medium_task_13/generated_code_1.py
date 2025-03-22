import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense
from keras import regularizers
from keras.optimizers import Adam

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv1, input_layer]))
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv2, conv1]))

    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv3, conv2]))
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv4, conv3]))
    bath_norm = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
    
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = keras.layers.Dropout(0.2)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    return model