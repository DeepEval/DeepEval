import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers import DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 convolutional layer
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Dropout after convolutional layers
    dropout1 = Dropout(0.25)(conv_1x1)
    
    # Second depthwise separable convolutional layer
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    
    # 1x1 convolutional layer
    conv_1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv2)
    
    # Dropout after convolutional layers
    dropout2 = Dropout(0.25)(conv_1x1_2)
    
    # Flatten the output
    flatten_layer = Flatten()(dropout2)
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model