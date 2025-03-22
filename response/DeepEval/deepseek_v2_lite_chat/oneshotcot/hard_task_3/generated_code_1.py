import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Process each group through a sequence of 1x1 convolution, 3x3 convolution, and dropout
    def process_group(group):
        conv1 = Conv2D(32, (1, 1), activation='relu')(group)
        conv3 = Conv2D(32, (3, 3), activation='relu')(conv1)
        drop = Dropout(0.2)(conv3)
        return drop
    
    processed_groups = [process_group(group) for group in split]
    
    # Concatenate outputs from the three groups
    concat = Concatenate()(processed_groups)
    
    # Main pathway: process the input through a 1x1 convolution to match output dimension
    def main_pathway(input_tensor):
        conv1x1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
        return conv1x1
    
    main_pathway_output = main_pathway(concat)
    bn = BatchNormalization()(main_pathway_output)
    flatten = Flatten()(bn)
    
    # Branch pathway: process the input through a 1x1 convolution
    branch_pathway_output = Conv2D(64, (1, 1), activation='relu')(concat)
    
    # Combine outputs from main and branch pathways using addition
    combined = Add()([main_pathway_output, branch_pathway_output])
    
    # Fully connected layer for classification
    dense = Dense(10, activation='softmax')(combined)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)
    
    return model