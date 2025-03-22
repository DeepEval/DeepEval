import keras
from keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseConv2D, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise Separable Convolution Layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 Convolution Layer for Feature Extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(depthwise_conv)
    
    # Dropout to mitigate overfitting
    dropout1 = Dropout(0.25)(conv1x1)
    
    # Another Depthwise Separable Convolution Layer
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    
    # 1x1 Convolution Layer for Feature Extraction
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(depthwise_conv2)
    
    # Dropout to mitigate overfitting
    dropout2 = Dropout(0.25)(conv1x1_2)
    
    # Flatten the output
    flatten = Flatten()(dropout2)
    
    # Fully Connected Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model