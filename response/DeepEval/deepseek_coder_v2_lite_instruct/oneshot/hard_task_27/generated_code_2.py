import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise Separable Convolution Layer
    conv_depthwise = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(input_layer)
    conv_depthwise = BatchNormalization()(conv_depthwise)
    conv_depthwise = ReLU()(conv_depthwise)
    
    # Fully Connected Layers for Channel-wise Feature Transformation
    fc1 = Dense(units=32, activation='relu')(conv_depthwise)
    fc2 = Dense(units=32, activation='relu')(fc1)
    
    # Addition Operation to Combine Original Input with Processed Features
    added = Add()([conv_depthwise, fc2])
    
    # Flatten Layer
    flatten_layer = Flatten()(added)
    
    # Final Two Fully Connected Layers for Classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model