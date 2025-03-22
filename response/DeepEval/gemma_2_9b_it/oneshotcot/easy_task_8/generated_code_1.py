import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():     

        input_layer = Input(shape=(28, 28, 1))
        
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
        dropout1 = Dropout(0.25)(depthwise) # Add dropout after depthwise conv

        pointwise = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
        dropout2 = Dropout(0.25)(pointwise) # Add dropout after pointwise conv

        flatten_layer = Flatten()(dropout2)
        output_layer = Dense(units=10, activation='softmax')(flatten_layer)

        model = keras.Model(inputs=input_layer, outputs=output_layer)

        return model