import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D
from keras.layers import Add
from keras import regularizers

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        split_axis = -1  # Split along the channel axis
        groups = 3
        channels = input_tensor.shape[-1]
        channels_per_group = channels // groups
        
        split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=groups, axis=split_axis))(input_tensor)
        
        conv1_group1 = Conv2D(filters=channels // groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv1_group2 = Conv2D(filters=channels // groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv1_group3 = Conv2D(filters=channels // groups, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        
        concat_conv1 = Concatenate()([conv1_group1, conv1_group2, conv1_group3])
        
        return concat_conv1
    
    conv1 = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        
        reshape_tensor = Reshape((height, width, groups, channels_per_group))(input_tensor)
        
        permute_tensor = Permute((1, 2, 4, 3))(reshape_tensor)
        
        reshape_permute_tensor = Reshape((height, width, channels))(permute_tensor)
        
        return reshape_permute_tensor
    
    conv2 = block2(conv1)
    
    # Block 3
    def block3(input_tensor):
        conv3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        return conv3
    
    conv3 = block3(conv2)
    
    # Branch
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Main path
    main_path = Add()([conv3, branch])
    
    # Fully connected layer
    flatten_layer = Flatten()(main_path)
    dense = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model