import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, AveragePooling2D, Concatenate, Lambda, DepthwiseConv2D, Flatten, Activation
from keras.models import Model

def dl_model():
    def block1(input_tensor):
        # Split input into three groups
        def split(x):
            return tf.split(x, num_or_size_splits=3, axis=-1)

        splits = Lambda(split)(input_tensor)

        # Apply 1x1 convolution to each split and reduce channel dimension
        convs = [Conv2D(filters=x.shape[-1]//3, kernel_size=(1, 1), activation='relu')(x) for x in splits]
        
        # Concatenate the outputs
        output = Concatenate()(convs)
        return output

    def block2(input_tensor):
        # Obtain the shape of input_tensor
        input_shape = tf.shape(input_tensor)

        # Reshape to (height, width, groups, channels_per_group)
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]
        reshaped = tf.reshape(input_tensor, (height, width, 3, channels // 3))

        # Permute dimensions for channel shuffling
        permuted = tf.transpose(reshaped, perm=(0, 1, 2, 3, 4))

        # Reshape back to original shape
        output = tf.reshape(permuted, (height, width, channels))
        
        return output

    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = block1(input_layer)
    main_path = block2(main_path)
    main_path = block3(main_path)
    main_path = block1(main_path)  # Repeat Block 1

    # Branch path using average pooling
    branch_path = AveragePooling2D(pool_size=(4, 4))(input_layer)

    # Concatenate main path and branch path
    concatenated = Concatenate()([main_path, branch_path])

    # Flatten and dense layer for classification
    flatten_layer = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()