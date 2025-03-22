import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Reshape, Lambda, Concatenate
from keras.layers import SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block: Average Pooling with different scales
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    flat1 = Flatten()(avg_pool1)
    
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    flat2 = Flatten()(avg_pool2)
    
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    flat3 = Flatten()(avg_pool3)
    
    concat_pools = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer and reshape
    fc_layer = Dense(units=256, activation='relu')(concat_pools)
    reshape_layer = Reshape((4, 4, 16))(fc_layer)  # Assuming a reshape that fits the next block's input
    
    # Second Block: Depthwise Separable Convolution
    def split_and_convolve(input_tensor):
        # Split along the last dimension into 4 groups
        splits = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        
        # Apply SeparableConv2D to each split
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(splits[3])
        
        # Concatenate the outputs
        return Concatenate()([conv1, conv2, conv3, conv4])
    
    processed_splits = Lambda(split_and_convolve)(reshape_layer)
    
    # Flatten and fully connected layer for classification
    flatten = Flatten()(processed_splits)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model