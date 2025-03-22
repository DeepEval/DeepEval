import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    conv = DepthwiseConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # 1x1 convolutional layer for feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv)
    
    # Dropout layer to mitigate overfitting
    drop1 = Dropout(0.2)(conv1x1)
    
    # Another 1x1 convolutional layer for feature extraction
    conv1x1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(drop1)
    
    # Dropout layer to mitigate overfitting
    drop2 = Dropout(0.2)(conv1x1_2)
    
    # Flatten the output
    flatten_layer = Flatten()(drop2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model