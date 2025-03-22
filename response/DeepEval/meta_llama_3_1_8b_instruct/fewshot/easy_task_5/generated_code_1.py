import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Apply a 1x1 convolution to reduce the dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Apply a 3x3 convolution to extract features
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Restore the dimensionality of the feature map using a 1x1 convolution layer
    conv3 = Conv2DTranspose(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the output
    flatten_layer = Flatten()(conv3)
    
    # Apply a fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model