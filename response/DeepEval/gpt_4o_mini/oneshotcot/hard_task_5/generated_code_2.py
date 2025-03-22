import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, Dense, Add, Reshape
import tensorflow as tf

def channel_shuffle(x, groups):
    # Get the batch size, height, width, and number of channels
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    channels = x.shape[-1]

    # Reshape to (batch_size, height, width, groups, channels/groups)
    x = tf.reshape(x, (batch_size, height, width, groups, channels // groups))
    
    # Transpose to (batch_size, height, width, channels/groups, groups)
    x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
    
    # Reshape back to (batch_size, height, width, channels)
    x = tf.reshape(x, (batch_size, height, width, channels))
    return x

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 image shape

    # Block 1
    split_features = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_features[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_features[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_features[2])
    block1_output = Concatenate(axis=-1)([conv1, conv2, conv3])  # Concatenate along channel dimension

    # Block 2
    groups = 3
    channels = block1_output.shape[-1]  # Number of channels from Block 1
    reshaped = Reshape((32, 32, groups, channels // groups))(block1_output)  # Reshape for channel shuffling
    shuffled = Lambda(lambda x: channel_shuffle(x, groups))(reshaped)  # Channel shuffle
    block2_output = Reshape((32, 32, channels))(shuffled)  # Reshape back to original shape

    # Block 3
    depthwise_conv = Conv2D(filters=channels, kernel_size=(3, 3), padding='same', groups=channels)(block2_output)  # Depthwise separable convolution

    # Branch connection
    branch_output = Add()([depthwise_conv, input_layer])  # Residual connection
    flatten_output = tf.keras.layers.Flatten()(branch_output)  # Flatten for fully connected layer
    dense_output = Dense(units=10, activation='softmax')(flatten_output)  # Output layer with softmax for classification

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model