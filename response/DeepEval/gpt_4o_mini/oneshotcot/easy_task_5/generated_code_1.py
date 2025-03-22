import keras
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolution layer for dimensionality reduction
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 convolution layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Restore the dimensionality with another 1x1 convolution layer
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv2)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv3)
    
    # Step 6: Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 7: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model