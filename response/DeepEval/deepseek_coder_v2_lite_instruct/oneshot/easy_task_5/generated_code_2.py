import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 convolution to reduce dimensionality
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer to extract features
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Second 1x1 convolution to restore dimensionality of the feature map
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the output
    flatten_layer = Flatten()(conv3)
    
    # Fully connected layer for classification
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model