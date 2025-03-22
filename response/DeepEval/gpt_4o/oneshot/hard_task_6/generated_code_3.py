import keras
from keras.layers import Input, Conv2D, Dense, AveragePooling2D, Concatenate, Flatten, Lambda, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    def block1(input_tensor):
        # Split the input into three parts along the channel dimension
        splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply a 1x1 convolution to each split
        convs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        
        # Concatenate the convolved outputs
        return Concatenate()(convs)
    
    def block2(input_tensor):
        # Reshape to (height, width, groups, channels_per_group)
        height, width, channels = input_tensor.shape[1:]
        reshaped = Lambda(lambda x: tf.reshape(x, [-1, height, width, 3, channels // 3]))(input_tensor)
        
        # Swap the groups and channels_per_group dimensions
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 2, 4, 3]))(reshaped)
        
        # Reshape back to the original shape
        return Lambda(lambda x: tf.reshape(x, [-1, height, width, channels]))(permuted)
    
    def block3(input_tensor):
        # Apply a 3x3 depthwise separable convolution
        return DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Repeated Block 1

    # Branch Path
    branch = AveragePooling2D(pool_size=(8, 8))(input_layer)
    branch = Flatten()(branch)

    # Concatenate main path and branch path outputs
    main_path_flatten = Flatten()(x)
    concatenated = Concatenate()([main_path_flatten, branch])
    
    # Fully Connected Layer for classification
    output_layer = Dense(units=10, activation='softmax')(concatenated)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model