import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    def split_channels(input_tensor):
        return tf.split(value=input_tensor, num_or_size_splits=3, axis=-1)
    
    groups = Lambda(split_channels)(input_layer)
    
    def process_group(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        dropout = Dropout(rate=0.5)(conv2)
        return dropout
    
    # Process each group through convolutions and dropout
    processed_groups = [process_group(group) for group in groups]
    
    # Concatenate results from the three groups
    main_pathway = Concatenate()(processed_groups)
    
    # Branch pathway: 1x1 convolution to match dimensions
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Combine pathways using an addition operation
    combined_pathways = Add()([main_pathway, branch_pathway])
    
    # Classification with a fully connected layer
    flatten_layer = Flatten()(combined_pathways)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model