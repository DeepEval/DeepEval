import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Dropout, Concatenate, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group with a sequence of 1x1 and 3x3 convolutions followed by a dropout layer
    def process_group(input_tensor):
        conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1x1)
        dropout = Dropout(rate=0.5)(conv3x3)
        return dropout
    
    # Apply the above processing to each group
    processed_groups = [process_group(group) for group in split_channels]
    
    # Concatenate the outputs from the three groups
    main_pathway = Concatenate()(processed_groups)
    
    # Branch pathway with a 1x1 convolution
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Combine main and branch pathways through addition
    combined_output = Add()([main_pathway, branch_pathway])
    
    # Flatten the output and add a fully connected layer for classification
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Construct and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model