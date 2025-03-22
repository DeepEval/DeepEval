import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Increase dimensionality of channels to 3x the original
    conv1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Depthwise separable convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    dw_conv = BatchNormalization()(dw_conv)
    dw_conv = Activation('relu')(dw_conv)
    
    # Channel attention
    avg_pool = GlobalAveragePooling2D()(dw_conv)
    fc1 = Dense(units=dw_conv.shape[channels] * 2, activation='relu')(avg_pool)
    activation = Dense(units=dw_conv.shape[channels])(fc1)
    attention = Activation('sigmoid')(activation)
    
    # Element-wise multiplication with the depthwise separable convolution
    weighted_conv = Multiply()([dw_conv, attention])
    
    # Reduce dimensionality with a 1x1 convolution
    conv2 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_conv)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    # Flatten and fully connected layers
    flatten = Flatten()(conv2)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()