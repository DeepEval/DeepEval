import keras
from keras.layers import Input, Conv2D, SeparableConv2D, AveragePooling2D, Dense, Concatenate, Lambda, Activation, Flatten
import tensorflow as tf

def dl_model():
    def block1(input_tensor):
        # Split the input into 3 parts
        split1, split2, split3 = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        
        # 1x1 convolution on each split
        conv1 = Conv2D(filters=split1.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split1)
        conv2 = Conv2D(filters=split2.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split2)
        conv3 = Conv2D(filters=split3.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split3)
        
        # Concatenate the outputs
        output = Concatenate()([conv1, conv2, conv3])
        return output

    def block2(input_tensor):
        # Get the shape
        shape = tf.shape(input_tensor)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 3
        channels_per_group = channels // groups
        
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = tf.reshape(input_tensor, [-1, height, width, groups, channels_per_group])
        
        # Permute the dimensions to swap third and fourth dimensions
        permuted = tf.transpose(reshaped, [0, 1, 2, 4, 3])
        
        # Reshape back to the original shape
        shuffled = tf.reshape(permuted, [-1, height, width, channels])
        return shuffled

    def block3(input_tensor):
        # Depthwise separable convolution
        depthwise_conv = SeparableConv2D(filters=input_tensor.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)

    # Branch Path
    branch = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch = Flatten()(branch)

    # Concatenate main and branch paths
    main_path = Flatten()(x)
    concatenated = Concatenate()([main_path, branch])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(concatenated)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model