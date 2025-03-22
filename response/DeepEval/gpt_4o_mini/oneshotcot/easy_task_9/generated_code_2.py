import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 with 1 channel (grayscale)
    
    # Step 2: Add 1x1 convolutional layer to increase dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1)
    
    # Step 4: Add another 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduction = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
    
    # Step 5: Add the output from the 1x1 reduction layer to the original input layer
    added_output = Add()([conv1x1_reduction, input_layer])
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(added_output)
    
    # Step 7: Add dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for digits 0-9
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model