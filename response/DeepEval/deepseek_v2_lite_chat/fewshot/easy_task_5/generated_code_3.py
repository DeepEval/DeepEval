import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Reduce dimensionality with a 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer to extract features
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Restore dimensionality with another 1x1 convolution
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the output
    flatten = Flatten()(conv3)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and return the model
model = dl_model()