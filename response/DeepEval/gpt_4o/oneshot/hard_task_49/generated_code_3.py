import keras
from keras.layers import Input, AveragePooling2D, Flatten, Dense, Concatenate, Reshape, Lambda, DepthwiseConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)
    
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer after first block
    dense1 = Dense(units=512, activation='relu')(concatenated)
    
    # Reshape to 4D tensor for input to second block
    reshape = Reshape((7, 7, 128))(dense1)  # 7x7 is chosen to match dimensions, 128 is arbitrary for illustration
    
    # Second block
    def split_and_process(input_tensor):
        split_tensors = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        
        path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        path4 = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(split_tensors[3])
        
        concatenated = Concatenate()([path1, path2, path3, path4])
        return concatenated
    
    processed = Lambda(split_and_process)(reshape)
    
    # Final layers
    flatten_final = Flatten()(processed)
    output_layer = Dense(units=10, activation='softmax')(flatten_final)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model