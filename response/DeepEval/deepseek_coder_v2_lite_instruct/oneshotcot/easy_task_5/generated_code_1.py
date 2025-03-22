import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolution layer
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 convolutional layer
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    
    # Step 4: Add 1x1 convolution layer to restore dimensionality
    conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv1x1_2)
    
    # Step 6: Add fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 7: Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model