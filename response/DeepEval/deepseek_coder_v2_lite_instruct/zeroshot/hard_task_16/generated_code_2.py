import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Flatten, Reshape

def dl_model():
    # Block 1
    def block_1(input_layer):
        # Split the input into three groups
        split_1, split_2, split_3 = tf.split(input_layer, num_or_size_splits=3, axis=-1)
        
        # Process each group through a series of convolutions
        conv_1_1 = Conv2D(32, (1, 1), activation='relu')(split_1)
        conv_3_3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_1)
        conv_1_2 = Conv2D(32, (1, 1), activation='relu')(conv_3_3)
        
        conv_1_3 = Conv2D(32, (1, 1), activation='relu')(split_2)
        conv_3_4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_3)
        conv_1_4 = Conv2D(32, (1, 1), activation='relu')(conv_3_4)
        
        conv_1_5 = Conv2D(32, (1, 1), activation='relu')(split_3)
        conv_3_6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv_1_5)
        conv_1_6 = Conv2D(32, (1, 1), activation='relu')(conv_3_6)
        
        # Concatenate the outputs from the three groups
        concatenated = Concatenate(axis=-1)([conv_1_2, conv_1_4, conv_1_6])
        
        return concatenated
    
    # Transition Convolution
    def transition_convolution(input_layer):
        # Adjust the number of channels to match the input layer
        conv = Conv2D(64, (1, 1), activation='relu')(input_layer)
        pooled = MaxPooling2D((2, 2), strides=(2, 2))(conv)
        return pooled
    
    # Block 2
    def block_2(input_layer):
        # Perform global max pooling
        pooled = GlobalMaxPooling2D()(input_layer)
        
        # Generate channel-matching weights through two fully connected layers
        fc_1 = Dense(128, activation='relu')(pooled)
        fc_2 = Dense(input_layer.shape[-1])(fc_1)
        
        # Reshape the weights to match the shape of the adjusted output
        reshaped = Reshape((1, 1, input_layer.shape[-1]))(fc_2)
        
        # Multiply the weights with the adjusted output
        weighted = tf.multiply(input_layer, reshaped)
        
        return weighted
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = block_1(input_layer)
    
    # Transition Convolution
    transition_output = transition_convolution(block1_output)
    
    # Block 2
    block2_output = block_2(transition_output)
    
    # Branch connecting directly to the input
    branch_output = input_layer
    
    # Add the outputs from both the main path and the branch
    added_output = tf.add(block2_output, branch_output)
    
    # Flatten the output and feed through a fully connected layer for classification
    flattened = Flatten()(added_output)
    output_layer = Dense(10, activation='softmax')(flattened)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()