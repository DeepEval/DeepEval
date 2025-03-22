import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # 7x7 depthwise separable convolutional layer
    conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', depthwise_constraint=None, activation='relu')(input_layer)
    conv = BatchNormalization()(conv)
    
    # Two fully connected layers with the same number of channels as the input layer
    flatten_layer = Flatten()(conv)
    dense1 = Dense(units=32, activation='relu')(flatten_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)
    
    # Addition operation to combine the original input with the processed features
    added = Add()([conv, dense2])
    
    # Final two fully connected layers for classification
    output_layer = Dense(units=10, activation='softmax')(added)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model