import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    def process_group(group):
        conv1x1 = Conv2D(32, (1, 1), activation='relu')(group)
        conv3x3 = Conv2D(32, (3, 3), activation='relu')(conv1x1)
        dropout = Dropout(0.5)(conv3x3)
        return dropout
    
    # Process each group
    processed_groups = [process_group(group) for group in split]
    
    # Concatenate the outputs from the three groups
    concatenated = Concatenate(axis=-1)(processed_groups)
    
    # Parallel branch with a 1x1 convolution to match the output dimension
    branch = Conv2D(64, (1, 1), activation='relu')(input_layer)
    
    # Addition operation to combine the outputs from both pathways
    combined = tf.add(concatenated, branch)
    
    # Flatten the combined output
    flattened = Flatten()(combined)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model