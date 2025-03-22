import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate, GlobalMaxPooling2D, Reshape
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    def block_1(input_tensor):
        # Split the input into three groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group through a series of convolutions
        conv_group1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[0])
        conv_group2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_group1)
        conv_group3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv_group2)
        
        conv_group4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[1])
        conv_group5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_group4)
        conv_group6 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv_group5)
        
        conv_group7 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_groups[2])
        conv_group8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_group7)
        conv_group9 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv_group8)
        
        # Concatenate the outputs from the three groups
        concatenated = Concatenate(axis=-1)([conv_group3, conv_group6, conv_group9])
        return concatenated

    block1_output = block_1(input_layer)

    # Transition Convolution to adjust the number of channels
    transition_conv = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(block1_output)

    # Block 2 with Global Max Pooling and reshaping weights
    block2 = GlobalMaxPooling2D()(transition_conv)
    fc_layer1 = Dense(units=128, activation='relu')(block2)
    fc_layer2 = Dense(units=64, activation='relu')(fc_layer1)
    
    # Generate weights for reshaping
    weights = Dense(units=64, activation='relu')(fc_layer2)
    reshaped_weights = Reshape((64, 1, 1))(weights)
    
    # Multiply weights with the transition_conv output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(transition_conv)
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_path_output)
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(main_path_output)
    
    # Branch directly from the input
    branch_output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Add the main path and branch outputs
    added_output = Add()([main_path_output, branch_output])
    
    # Flatten and pass through fully connected layer for classification
    flattened_output = Flatten()(added_output)
    final_output = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=final_output)
    return model