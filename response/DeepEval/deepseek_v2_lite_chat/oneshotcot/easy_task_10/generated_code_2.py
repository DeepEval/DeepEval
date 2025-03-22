import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First 1x1 Convolution layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Depthwise Separable Convolution layer for feature extraction
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    dw_conv_bn = BatchNormalization()(dw_conv)
    
    # Reduce dimensionality using 1x1 Convolution with stride 2
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(dw_conv_bn)
    
    # Concatenate the outputs of the first and second convolutional layers
    concat = Concatenate()([conv1, conv2])
    
    # Block of four parallel paths
    def block(input_tensor):
        # 1x1 Convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 Convolution
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 5x5 Convolution
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 1x1 MaxPooling
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Concatenate outputs of all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output = block(concat)
    # Add batch normalization and flatten the result
    batch_norm = BatchNormalization()(block_output)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])