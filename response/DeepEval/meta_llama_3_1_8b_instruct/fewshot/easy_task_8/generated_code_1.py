import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 convolutional layer for feature extraction
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv)
    
    # Dropout layer to mitigate overfitting
    dropout = Dropout(0.2)(conv)
    
    # Another dropout layer to mitigate overfitting
    dropout2 = Dropout(0.2)(dropout)
    
    # Flatten the output
    flatten = Flatten()(dropout2)
    
    # Output layer with a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model