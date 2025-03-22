import keras
from keras.layers import Input, Lambda, Split, SeparableConv2D, Concatenate, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input along the channel axis into three groups
    input_split = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Define the first block
    first_block = []
    for i in range(3):
        # Apply separable convolution with different kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_split[i])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_split[i])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_split[i])
        # Concatenate the outputs from each group
        first_block.append(Concatenate()([conv1, conv2, conv3]))

    # Define the second block
    second_block = []
    for i in range(3):
        # Apply a 3x3 convolution
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_split[i])
        # Apply a series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_split[i])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)
        # Apply a max pooling branch
        conv3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
        # Concatenate the outputs from each branch
        second_block.append(Concatenate()([conv1, conv2, conv3]))

    # Define the global average pooling layer
    global_average_pooling = GlobalAveragePooling2D()(Concatenate()(first_block + second_block))

    # Define the fully connected layer
    fully_connected = Dense(units=10, activation='softmax')(global_average_pooling)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fully_connected)

    return model