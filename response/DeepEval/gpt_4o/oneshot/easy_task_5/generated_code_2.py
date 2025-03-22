import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Reduce the input dimensionality with a 1x1 convolution
    conv1x1_reduce = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Extract features using a 3x3 convolutional layer
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1_reduce)
    
    # Restore the dimensionality with another 1x1 convolution layer
    conv1x1_restore = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3x3)
    
    # Flatten the output
    flatten_layer = Flatten()(conv1x1_restore)
    
    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model