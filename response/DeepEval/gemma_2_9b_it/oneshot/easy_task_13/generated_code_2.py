import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
        dropout1 = Dropout(0.25)(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(0.25)(conv2)
        conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout2)
        dropout3 = Dropout(0.25)(conv3)
        conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout3)
        
        conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4) 
        
        output = Add()([input_layer, conv5])
        
        flatten_layer = Flatten()(output)
        dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=dense_layer)

        return model