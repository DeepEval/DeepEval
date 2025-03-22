import keras
from keras.layers import Input, Conv2D, Lambda, Reshape, Permute, Add, Dense, DepthwiseConv2D
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block_1(input_tensor):
        split = Lambda(lambda x: K.tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[2])
        output = Concatenate()([conv1, conv2, conv3])
        return output
    
    block1_output = block_1(input_layer)
    
    # Block 2
    def block_2(input_tensor):
        feature_shape = K.int_shape(input_tensor)
        reshaped = Reshape((feature_shape[1], feature_shape[2], 3, int(feature_shape[-1]/3)))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        reshaped_back = Reshape(feature_shape)(permuted)
        return reshaped_back
    
    block2_output = block_2(block1_output)
    
    # Block 3
    def block_3(input_tensor):
        output = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output
    
    block3_output = block_3(block2_output)
    
    # Branch
    branch_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch
    combined = Add()([block3_output, branch_output])
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(combined)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model