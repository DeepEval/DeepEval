import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    def block1(input_tensor):
        # Split the input tensor into three groups
        split_tensor = Lambda(lambda x: K.split(x, 3, axis=-1))(input_tensor)
        group1 = Conv2D(filters=32//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        group2 = Conv2D(filters=32//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        group3 = Conv2D(filters=32//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor
    
    def block2(input_tensor):
        # Get the shape of the input tensor
        shape = K.shape(input_tensor)
        # Reshape the input tensor to (height, width, groups, channels_per_group)
        reshaped_tensor = Reshape((shape[1], shape[2], shape[3]//3, 3))(input_tensor)
        # Swap the third and fourth dimensions
        permuted_tensor = Permute((2, 3, 1, 4))(reshaped_tensor)
        # Reshape the input tensor back to its original shape
        output_tensor = Reshape((shape[1], shape[2], shape[3]))(permuted_tensor)
        return output_tensor
    
    def block3(input_tensor):
        # Apply a 3x3 depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor
    
    # Block 1
    conv1 = block1(input_layer)
    bath_norm1 = BatchNormalization()(conv1)
    
    # Block 2
    conv2 = block2(bath_norm1)
    bath_norm2 = BatchNormalization()(conv2)
    
    # Block 3
    conv3 = block3(bath_norm2)
    bath_norm3 = BatchNormalization()(conv3)
    
    # Block 1 (repeated)
    conv4 = block1(bath_norm3)
    bath_norm4 = BatchNormalization()(conv4)
    
    # Average pooling layer
    avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(input_layer)
    avg_pool = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(avg_pool)
    avg_pool = BatchNormalization()(avg_pool)
    
    # Concatenate the outputs from the main path and the branch path
    output_tensor = Concatenate()([bath_norm4, avg_pool])
    
    # Flatten the combined output
    flatten_layer = Flatten()(output_tensor)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model