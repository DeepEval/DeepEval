import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Lambda, SeparableConv2D, BatchNormalization
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)
    
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    concat_pools = Concatenate()([flat1, flat2, flat3])
    
    # Dense and Reshape between first and second block
    dense1 = Dense(units=256, activation='relu')(concat_pools)
    reshaped = Reshape(target_shape=(4, 4, 16))(dense1)  # Assuming output shape matches
    
    # Second Block
    def split_block(input_tensor):
        # Split into four groups along the last dimension
        split_tensors = tf.split(input_tensor, num_or_size_splits=4, axis=-1)
        
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_tensors[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_tensors[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_tensors[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split_tensors[3])
        
        concat_convs = Concatenate()([conv1, conv2, conv3, conv4])
        
        return concat_convs
    
    processed_block = Lambda(split_block)(reshaped)
    batch_norm = BatchNormalization()(processed_block)
    flattened = Flatten()(batch_norm)
    
    # Final Dense Layer for Classification
    output_layer = Dense(units=10, activation='softmax')(flattened)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model