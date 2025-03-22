import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def block(input_tensor):
        # 1x1 convolution to increase the depth
        conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Depthwise separable convolution
        dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        # 1x1 convolution to reduce the depth back to the original
        conv1x1_reduced = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(dw_conv)
        # MaxPooling2D to reduce spatial dimensions
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Concatenate the results from different paths
        concat_tensor = Concatenate(axis=-1)([conv1x1, dw_conv, conv1x1_reduced, max_pool])
        
        return concat_tensor

    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Three branches, each with the block
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate the outputs from all branches
    concat_output = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Flatten the concatenated tensor
    flat_output = Flatten()(concat_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flat_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])