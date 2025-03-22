import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise Separable Convolution Layer
    depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', depthwise_constraint=None, activation='relu')(input_layer)
    depthwise_conv = Dropout(0.2)(depthwise_conv)
    
    # 1x1 Convolutional Layer for Feature Extraction
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    conv1x1 = Dropout(0.2)(conv1x1)
    
    # Flatten the output
    flatten_layer = Flatten()(conv1x1)
    
    # Fully Connected Layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model