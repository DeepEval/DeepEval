import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer
    conv2 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)
    
    # Flatten layer
    flatten = Flatten()(conv3)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model