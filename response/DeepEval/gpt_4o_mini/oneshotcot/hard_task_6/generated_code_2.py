import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda
from keras.models import Model

def split_channels(input_tensor):
    # Split the tensor into three parts along the channel axis
    return tf.split(input_tensor, num_or_size_splits=3, axis=-1)

def channel_shuffle(input_tensor, groups):
    # Get the input shape
    batch_size, height, width, channels = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], tf.shape(input_tensor)[3]
    channels_per_group = channels // groups
    
    # Reshape to (batch_size, height, width, groups, channels_per_group)
    x = tf.reshape(input_tensor, [batch_size, height, width, groups, channels_per_group])
    
    # Permute the dimensions to shuffle the channels
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    
    # Reshape back to the original shape
    return tf.reshape(x, [batch_size, height, width, channels])

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split = Lambda(split_channels)(input_layer)
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split[0])
    conv2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split[1])
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(split[2])
    block1_output = Concatenate()([conv1, conv2, conv3])

    # Block 2
    reshaped_block1 = tf.reshape(block1_output, (-1, 32, 32, 3, 16))
    shuffled_output = channel_shuffle(reshaped_block1, groups=3)

    # Block 3
    block3_output = Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu', depthwise=True)(shuffled_output)

    # Main path output after Block 3
    main_path_output = block3_output

    # Branch path (Average pooling)
    branch_path_output = AveragePooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate main path and branch path
    combined_output = Concatenate()([main_path_output, branch_path_output])

    # Fully connected layer
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model