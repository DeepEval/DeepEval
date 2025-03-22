import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Dense, Add, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input along the channel dimension into three parts
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def process_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        dropout = Dropout(rate=0.5)(conv3)
        return dropout
    
    # Process each split group
    group1_output = process_group(split_layer[0])
    group2_output = process_group(split_layer[1])
    group3_output = process_group(split_layer[2])
    
    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()([group1_output, group2_output, group3_output])
    
    # Branch pathway: process the input through a 1x1 convolution
    branch_pathway = Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from main and branch pathways
    combined_output = Add()([main_pathway, branch_pathway])
    
    # Flatten the result and add a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model