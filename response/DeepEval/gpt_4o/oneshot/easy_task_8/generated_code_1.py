import keras
from keras.layers import Input, SeparableConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolution layer
    sep_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    dropout1 = Dropout(rate=0.3)(sep_conv)
    
    # 1x1 convolutional layer for feature extraction
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.3)(conv_1x1)
    
    # Flatten the feature map and add a dense layer for classification
    flatten_layer = Flatten()(dropout2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model