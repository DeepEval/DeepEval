import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Lambda, Reshape, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Average pooling layers with different scales
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the pooled outputs
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    # Concatenate the flattened outputs
    concat1 = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer and reshape
    dense1 = Dense(units=128, activation='relu')(concat1)
    reshaped = Reshape((1, 1, 128))(dense1)  # Reshape to (1, 1, 128) for next block

    # Second block: Depthwise separable convolutions with different kernel sizes
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped)
    
    # Apply depthwise separable convolutions
    depthwise_conv1 = DepthwiseConv2D(kernel_size=(1, 1), activation='relu', padding='same')(split_tensor[0])
    depthwise_conv2 = DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(split_tensor[1])
    depthwise_conv3 = DepthwiseConv2D(kernel_size=(5, 5), activation='relu', padding='same')(split_tensor[2])
    depthwise_conv4 = DepthwiseConv2D(kernel_size=(7, 7), activation='relu', padding='same')(split_tensor[3])

    # Concatenate the outputs of the depthwise convolutions
    concat2 = Concatenate()([depthwise_conv1, depthwise_conv2, depthwise_conv3, depthwise_conv4])

    # Flatten the concatenated outputs
    flat_output = Flatten()(concat2)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flat_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()