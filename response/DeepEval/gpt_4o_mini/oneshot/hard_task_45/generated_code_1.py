import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, Lambda, DepthwiseConv2D
import tensorflow as tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block: Splitting input into three groups and applying depthwise separable convolutions
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Convolution with different kernel sizes for each split
    path1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu', padding='same')(split_tensor[0])
    path2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(split_tensor[1])
    path3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu', padding='same')(split_tensor[2])

    # Concatenate the outputs of the first block
    block1_output = Concatenate()([path1, path2, path3])

    # Second block: Multiple branches for feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch2)
    
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(block1_output)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(branch3)
    
    branch4 = MaxPooling2D(pool_size=(2, 2), padding='same')(block1_output)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(branch4)

    # Concatenate the outputs of the second block
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and pass through a fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model