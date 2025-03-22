import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block with three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten each pooling output
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    # Concatenate the flattened outputs
    concat1 = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer followed by reshape
    dense1 = Dense(units=512, activation='relu')(concat1)
    reshape = Reshape((4, 4, 32))(dense1)  # Reshape to 4D tensor for the second block
    
    # Second block: split and depthwise separable convolutions
    def split_and_conv(input_tensor):
        # Split into 4 groups
        split_tensors = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        # Apply different depthwise separable convolutions
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split_tensors[3])
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    block2_output = split_and_conv(reshape)
    
    # Flatten the result from second block
    flatten_final = Flatten()(block2_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_final)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model