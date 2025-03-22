import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise Separable Convolution
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_constraint=None, bias_constraint=None)(input_layer)
    depthwise_conv = Dropout(0.25)(depthwise_conv)
    
    # 1x1 Convolutional Layer
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    conv_1x1 = Dropout(0.25)(conv_1x1)
    
    # Flatten the output
    flatten_layer = Flatten()(conv_1x1)
    
    # Fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model