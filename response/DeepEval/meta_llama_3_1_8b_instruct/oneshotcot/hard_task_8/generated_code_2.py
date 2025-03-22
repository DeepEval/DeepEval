import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv2)
    
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    depthwise_conv5 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv5)

    output_tensor = Concatenate()([conv3, conv6])
    
    block_output = output_tensor
    
    # Obtain the shape of the features from Block 1
    feature_shape = K.int_shape(block_output)
    
    # Reshape the features into four groups
    reshaped_features = Reshape((feature_shape[1], feature_shape[2], feature_shape[3] // 4, 4))(block_output)
    
    # Swap the third and fourth dimensions
    permuted_features = Permute((3, 1, 2, 4))(reshaped_features)
    
    # Reshape the features back to its original shape
    reshaped_permuted_features = Reshape(feature_shape)(permuted_features)
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(reshaped_permuted_features)
    
    # Flatten the features
    flatten_layer = Flatten()(bath_norm)
    
    # Add fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model