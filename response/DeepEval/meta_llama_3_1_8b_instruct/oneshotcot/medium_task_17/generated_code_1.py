import keras
from keras.layers import Input, Lambda, Reshape, Permute, Conv2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Get the shape of the input layer
    input_shape = input_layer.shape
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    
    # Reshape the input tensor into three groups
    group_shape = (height, width, 3, channels // 3)
    reshape_group = Reshape(group_shape)(input_layer)
    
    # Swap the third and fourth dimensions using a permutation operation
    # This is equivalent to channel shuffling in group convolution
    permute_channels = Permute((1, 2, 4, 3))(reshape_group)
    
    # Reshape the tensor back to its original input shape
    reshape_original = Reshape(input_shape)(permute_channels)
    
    # Add a fully connected layer with softmax activation
    output_layer = Dense(units=10, activation='softmax')(reshape_original)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model