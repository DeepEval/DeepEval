import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    depthwise_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.25)(depthwise_conv)
    
    # 1x1 convolutional layer
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.25)(conv_1x1)
    
    # Flatten layer
    flatten_layer = Flatten()(dropout2)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model