import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Reshape, Permute

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(depthwise_conv1)

    branch_conv1 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu')(input_layer)
    branch_conv2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch_conv1)
    
    merged_block1 = Concatenate(axis=-1)([conv1_2, branch_conv2])

    # Block 2
    shape_layer = Lambda(lambda x: tf.shape(x))(merged_block1)
    reshaped_features = Reshape(target_shape=(28, 28, 4, 128))(merged_block1)  
    permuted_features = Permute((0, 1, 3, 2))(reshaped_features)
    reshaped_back = Reshape(target_shape=(28, 28, 128))(permuted_features) 

    flatten = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model