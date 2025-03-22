import keras
from keras.layers import Input, Concatenate, Dense, Flatten, Lambda, DepthwiseConv2D
from keras.models import Model

# Define the input shape
input_shape = (32, 32, 3)

# Define the first block
def block_1(input_tensor):
    # Split the input into three groups along the last dimension
    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)

    # Apply depthwise separable convolutional layers with varying kernel sizes to each group
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])

    # Concatenate the outputs from all groups
    output = Concatenate()([conv1, conv2, conv3])

    return output

# Define the second block
def block_2(input_tensor):
    # Apply multiple branches for feature extraction
    branches = []

    # Branch 1: 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    branches.append(conv1)

    # Branch 2: <1x1 convolution, 3x3 convolution>
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
    branches.append(conv3)

    # Branch 3: <max pooling, 1x1 convolution>
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
    branches.append(conv4)

    # Concatenate the outputs from all branches
    output = Concatenate()(branches)

    return output

# Define the model
input_layer = Input(shape=input_shape)
output_layer = block_1(input_layer)
output_layer = block_2(output_layer)
output_layer = Flatten()(output_layer)
output_layer = Dense(units=10, activation='softmax')(output_layer)
model = Model(inputs=input_layer, outputs=output_layer)