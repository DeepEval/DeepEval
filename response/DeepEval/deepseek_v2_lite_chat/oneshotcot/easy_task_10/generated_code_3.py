import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, ZeroPadding2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolution to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # 3x3 depthwise separable convolution for feature extraction
    depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    depthwise_sep = keras.layers.Activation('relu')(depthwise)
    
    # 1x1 Convolution to reduce dimensionality with stride of 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same')(depthwise_sep)
    
    # Max Pooling with stride of 2 to match the stride of the Conv2D
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
    
    # Concatenate the outputs from the different paths
    concat = Concatenate()(inputs=[depthwise_sep, conv2, conv1, avg_pool])
    
    # Batch Normalization
    batch_norm = BatchNormalization()(concat)
    
    # Flatten the output
    flatten = Flatten()(batch_norm)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()