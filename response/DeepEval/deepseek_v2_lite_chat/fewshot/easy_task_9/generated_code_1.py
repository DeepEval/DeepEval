import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolution Layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 Depthwise Separable Convolution Layer
    dw_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    dw_conv1_act = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dw_conv1)
    
    # 1x1 Convolution Layer
    conv2 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Add operation to concatenate the outputs of the two paths
    add_layer = Add()([conv1, dw_conv1_act, conv2])
    
    # Flatten layer to convert the output tensor into a 1D vector
    flatten_layer = Flatten()(add_layer)
    
    # Fully Connected Layer for classification
    dense = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output Layer for final classification
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Model Creation
model = dl_model()
model.summary()