import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape
from tensorflow.keras.layers import Lambda, DepthwiseConv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: average pooling layers with varying scales
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool1_flat = Flatten()(pool1)
    
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool2_flat = Flatten()(pool2)
    
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    pool3_flat = Flatten()(pool3)
    
    # Concatenate the flattened pooling outputs
    concat_output = Concatenate()([pool1_flat, pool2_flat, pool3_flat])
    
    # Transform the output into a 4-dimensional tensor
    dense = Dense(128)(concat_output)
    reshape = Reshape((128, 1, 1))(dense)
    
    # Second block: depthwise separable convolutional layers
    def split_input(input_tensor):
        # Split the input along the last dimension
        outputs = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        return outputs
    
    split_layer = Lambda(split_input)(reshape)
    
    conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_layer[3])
    
    # Concatenate the convolutional outputs
    output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
    
    flatten_layer = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model