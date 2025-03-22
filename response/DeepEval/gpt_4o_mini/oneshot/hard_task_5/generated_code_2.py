import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Dense, Lambda, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Split into three groups and apply 1x1 convolution
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
    conv2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[1])
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[2])
    
    # Concatenate the outputs along the channel dimension
    block1_output = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Block 2: Channel shuffling
    shape = tf.shape(block1_output)
    reshaped = Reshape((shape[1], shape[2], 3, shape[3] // 3))(block1_output)  # Reshape to (height, width, groups, channels_per_group)
    permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2, 4]))(reshaped)  # Swap groups and channels
    block2_output = Reshape((shape[1], shape[2], shape[3]))(permuted)  # Reshape back to original shape

    # Block 3: Depthwise separable convolution
    depthwise_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', use_bias=False, groups=64)(block2_output)
    
    # Direct branch connecting to the input
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the main path and branch
    combined_output = Add()([depthwise_conv, branch_output])

    # Fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # To display the model architecture