import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: 1x1 convolution to reduce dimensionality
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: 3x3 convolution for feature extraction
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: 1x1 convolution to restore dimensionality
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv3)
    
    # Step 6: Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model