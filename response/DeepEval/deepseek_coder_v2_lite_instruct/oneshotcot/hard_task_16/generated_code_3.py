import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, GlobalMaxPooling2D, Add, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(x):
        # Split the input into three groups
        split1 = Lambda(lambda y: y[:, :, :, :y.shape[3]//3])(x)
        split2 = Lambda(lambda y: y[:, :, :, y.shape[3]//3:2*y.shape[3]//3])(x)
        split3 = Lambda(lambda y: y[:, :, :, 2*y.shape[3]//3:])(x)
        
        # Process each split through convolutions
        conv1_1 = Conv2D(64, (1, 1), activation='relu')(split1)
        conv3_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
        conv1_2 = Conv2D(64, (1, 1), activation='relu')(conv3_1)
        
        conv1_3 = Conv2D(64, (1, 1), activation='relu')(split2)
        conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_3)
        conv1_4 = Conv2D(64, (1, 1), activation='relu')(conv3_2)
        
        conv1_5 = Conv2D(64, (1, 1), activation='relu')(split3)
        conv3_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_5)
        conv1_6 = Conv2D(64, (1, 1), activation='relu')(conv3_3)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1_2, conv1_4, conv1_6])
        return concatenated
    
    block1_output = block1(input_layer)
    
    # Transition convolution
    transition_conv = Conv2D(32, (1, 1), activation='relu')(block1_output)
    
    # Block 2
    block2_input = transition_conv
    block2_output = GlobalMaxPooling2D()(block2_input)
    
    # Fully connected layers for generating weights
    fc1 = Dense(128, activation='relu')(block2_output)
    fc2 = Dense(block2_input.shape[3], activation='softmax')(fc1)
    
    # Reshape weights to match the shape of block2_input
    reshaped_weights = tf.reshape(fc2, (fc2.shape[0], 1, 1, fc2.shape[1]))
    
    # Multiply weights with block2_input
    main_path_output = tf.multiply(block2_input, reshaped_weights)
    
    # Direct branch from input
    branch = input_layer
    
    # Add the main path and the branch outputs
    added_output = Add()([main_path_output, branch])
    
    # Flatten the final output
    flattened_output = Flatten()(added_output)
    
    # Fully connected layer for classification
    output_layer = Dense(10, activation='softmax')(flattened_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()