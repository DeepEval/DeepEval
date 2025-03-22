import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Dropout, Add, Flatten, Dense
from keras import regularizers
import tensorflow as tf

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the main pathway
    def conv_block(x):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop = Dropout(0.2)(conv2)
        return drop
    
    # Apply the convolution block to each group
    group1 = conv_block(split_layer[0])
    group2 = conv_block(split_layer[1])
    group3 = conv_block(split_layer[2])
    
    # Concatenate the outputs from the three groups
    concat_layer = Concatenate()([group1, group2, group3])
    
    # Define the branch pathway
    branch_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both pathways using an addition operation
    add_layer = Add()([concat_layer, branch_layer])
    
    # Apply batch normalization
    bath_norm = BatchNormalization()(add_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Apply fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model