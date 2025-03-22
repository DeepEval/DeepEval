import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Add, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Define the convolutional blocks for each group
    def conv_block(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        return conv3x3
    
    # Apply the convolutional blocks to each group
    conv_blocks = [conv_block(group) for group in split_layer]
    
    # Concatenate the outputs from the three groups to form the main pathway
    main_pathway = Concatenate()(conv_blocks)
    
    # Parallel branch: Process the input through a 1x1 convolution to match the output dimension
    parallel_branch = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Addition operation to combine the main pathway and the parallel branch
    combined_output = Add()([main_pathway, parallel_branch])
    
    # Flatten the result
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dropout)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.summary()