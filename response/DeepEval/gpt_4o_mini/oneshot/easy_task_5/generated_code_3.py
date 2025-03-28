import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    
    # Input layer with shape for MNIST dataset (28x28 images with 1 channel)
    input_layer = Input(shape=(28, 28, 1))
    
    # Dimensionality reduction with a 1x1 convolution
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Feature extraction with a 3x3 convolution
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Restoring dimensionality with another 1x1 convolution
    conv1x1_restore = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3x3)
    
    # Flatten the output
    flatten_layer = Flatten()(conv1x1_restore)
    
    # Fully connected layer with 10 neurons for classification (softmax activation)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model