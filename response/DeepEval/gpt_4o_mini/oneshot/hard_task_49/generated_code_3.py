import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Concatenate, DepthwiseConv2D, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers with varying scales
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten each pooling output
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenate flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer after first block
    dense1 = Dense(units=128, activation='relu')(concatenated)

    # Reshape output for second block
    reshaped = Reshape(target_shape=(-1, 1, 128))(dense1)

    # Second block: Depthwise separable convolutions
    # Split the reshaped tensor into 4 groups
    split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)

    # Process each split with depthwise separable convolution
    depthwise_outputs = []
    for kernel_size in [1, 3, 5, 7]:
        depthwise_conv = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(split_tensors[0])
        depthwise_outputs.append(depthwise_conv)
    
    # Concatenate outputs from depthwise separable convolutions
    concatenated_depthwise = Concatenate()(depthwise_outputs)

    # Flatten and apply the final fully connected layer
    flatten_depthwise = Flatten()(concatenated_depthwise)
    output_layer = Dense(units=10, activation='softmax')(flatten_depthwise)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model