import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Lambda, Concatenate
from keras.layers import DepthwiseConv2D
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Average Pooling Layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten the outputs
    flat1 = Flatten()(avg_pool1)
    flat2 = Flatten()(avg_pool2)
    flat3 = Flatten()(avg_pool3)

    # Concatenate the flattened vectors
    concat_block1 = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer
    dense_layer = Dense(units=128, activation='relu')(concat_block1)
    
    # Reshape to 4D tensor for second block
    reshaped_block1 = Lambda(lambda x: tf.reshape(x, (-1, 4, 4, 9)))(dense_layer)

    # Second Block: Depthwise Separable Convolutions
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshaped_block1)

    # Apply Depthwise Separable Convolutions
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensor[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensor[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensor[2])
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_tensor[3])

    # Concatenate the outputs from depthwise convolutions
    concat_block2 = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Flatten the result and fully connected layer for classification
    flatten_block2 = Flatten()(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(flatten_block2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model