import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    separable_conv = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv1)
    
    # 1x1 convolutional layer to reduce dimensionality
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(separable_conv)
    
    # Flattening layer and fully connected layer for classification
    flatten_layer = Flatten()(conv2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model