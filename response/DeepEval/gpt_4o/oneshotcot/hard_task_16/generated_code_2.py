import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Dense, GlobalMaxPooling2D, Multiply, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split input into 3 groups and apply series of convolutions
    def block1(input_tensor):
        # Split into 3 groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Apply convolutions to each group
        convs = []
        for group in split_groups:
            x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(group)
            x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
            x = Conv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(x)
            convs.append(x)
        
        # Concatenate the outputs
        output_tensor = Concatenate()(convs)
        
        return output_tensor
    
    # Transition Convolution: Adjust the number of channels
    def transition_convolution(input_tensor):
        num_channels = input_layer.shape[-1]
        adjusted_output = Conv2D(filters=num_channels, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        return adjusted_output
    
    # Block 2: Global Max Pooling and channel matching weights
    def block2(input_tensor):
        # Global Max Pooling
        pooled_output = GlobalMaxPooling2D()(input_tensor)
        
        # Fully connected layers to generate channel-matching weights
        dense1 = Dense(units=128, activation='relu')(pooled_output)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        
        # Reshape weights
        weights = Lambda(lambda x: tf.reshape(x, (-1, 1, 1, input_tensor.shape[-1])))(dense2)
        
        # Multiply weights with adjusted output
        main_path_output = Multiply()([input_tensor, weights])
        
        return main_path_output
    
    # Implementing the architecture
    block1_output = block1(input_layer)
    trans_conv_output = transition_convolution(block1_output)
    block2_output = block2(trans_conv_output)
    
    # Branch that connects directly to the input
    branch_output = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)
    
    # Adding the main path output and the branch output
    add_output = Add()([block2_output, branch_output])
    
    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(GlobalMaxPooling2D()(add_output))
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model