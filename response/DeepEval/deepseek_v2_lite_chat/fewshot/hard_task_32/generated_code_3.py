import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Add, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    def branch1(input_tensor):
        # Depthwise separable convolution
        conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolution
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        # Dropout for regularization
        conv2 = Dropout(0.5)(conv2)
        return conv2
    
    branch1_output = branch1(input_tensor=input_layer)
    
    # Branch 2
    def branch2(input_tensor):
        # Depthwise separable convolution
        conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolution
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        # Dropout for regularization
        conv4 = Dropout(0.5)(conv4)
        return conv4
    
    branch2_output = branch2(input_tensor=input_layer)
    
    # Branch 3
    def branch3(input_tensor):
        # Depthwise separable convolution
        conv5 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 convolution
        conv6 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv5)
        # Dropout for regularization
        conv6 = Dropout(0.5)(conv6)
        return conv6
    
    branch3_output = branch3(input_tensor=input_layer)
    
    # Concatenate outputs from all branches
    concatenated = Add()([branch1_output, branch2_output, branch3_output])
    
    # Fully connected layers for classification
    flatten_layer = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the deep learning model
model = dl_model()