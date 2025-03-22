from keras.layers import Input, DepthwiseConv2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block of the model
    first_block = Input(shape=input_shape)

    # Split the input into three groups along the last dimension
    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(first_block)

    # Apply depthwise separable convolutional layers with varying kernel sizes to each group
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])

    # Concatenate the outputs from the three groups
    output = Concatenate()([conv1, conv2, conv3])

    # Flatten the output and add a fully connected layer for classification
    flatten = Flatten()(output)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Define the second block of the model
    second_block = Input(shape=input_shape)

    # Apply multiple branches for feature extraction
    branches = []
    for i in range(3):
        branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(second_block)
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch)
        branch = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch)
        branches.append(branch)

    # Concatenate the outputs from the three branches
    output = Concatenate()(branches)

    # Flatten the output and add a fully connected layer for classification
    flatten = Flatten()(output)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=[first_block, second_block], outputs=dense)

    return model