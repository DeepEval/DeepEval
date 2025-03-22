import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, DepthwiseConv2D, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Step 4: Add another 1x1 convolutional layer to reduce dimensionality and apply stride of 2
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(conv2)
    
    # Step 5: Flatten the result
    flatten_layer = Flatten()(conv3)
    
    # Step 6: Add a fully connected layer to produce the final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model