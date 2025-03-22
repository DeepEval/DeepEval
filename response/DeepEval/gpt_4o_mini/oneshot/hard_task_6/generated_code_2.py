import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, Reshape
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        # Split the input tensor into 3 groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # 1x1 convolution for each group
        convs = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(group) for group in groups]
        
        # Concatenate the outputs of the convolutions
        return Concatenate()(*convs)

    # Main path
    block1_output = block1(input_layer)
    block2_output = block1(block1_output)  # Repeating Block 1
    block3_output = block1(block2_output)  # Repeating Block 1 again

    # Block 2 - Channel Shuffling
    def block2(input_tensor):
        # Obtain the shape of the input tensor
        shape = tf.shape(input_tensor)
        height, width = shape[1], shape[2]
        
        # Reshape and permute the tensor
        reshaped = Reshape((height, width, 3, -1))(input_tensor)  # groups=3
        shuffled = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(reshaped)  # Swap the last two dimensions
        return Reshape(tf.shape(input_tensor)[1:])(shuffled)  # Reshape back to original

    block2_output = block2(block3_output)

    # Block 3 - Depthwise Separable Convolution
    block3_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', groups=3)(block2_output)

    # Branch path - Average pooling
    branch_output = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)

    # Concatenate main path and branch path
    concatenated_output = Concatenate()([block3_output, branch_output])

    # Fully connected layer for classification
    flatten_layer = Flatten()(concatenated_output)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=dense_output)

    return model