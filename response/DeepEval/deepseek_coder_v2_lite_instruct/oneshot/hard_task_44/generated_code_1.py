import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Feature extraction with different kernel sizes
    def block1(x):
        # Split the input into three groups
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # Feature extraction for each group
        conv1x1 = Conv2D(64, (1, 1), activation='relu')(splits[0])
        conv3x3 = Conv2D(64, (3, 3), activation='relu')(splits[1])
        conv5x5 = Conv2D(64, (5, 5), activation='relu')(splits[2])
        
        # Dropout to reduce overfitting
        dropout = Dropout(0.25)(conv5x5)
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, dropout])
        
        return concatenated
    
    # Apply block 1 to the input
    block1_output = block1(input_layer)
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(block1_output)
    flatten = Flatten()(batch_norm)
    
    # Block 2: Four branches for feature fusion
    def block2(x):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(64, (1, 1), activation='relu')(x)
        
        # Branch 2: <1x1 convolution, 3x3 convolution>
        branch2a = Conv2D(64, (1, 1), activation='relu')(x)
        branch2b = Conv2D(64, (3, 3), activation='relu')(branch2a)
        
        # Branch 3: <1x1 convolution, 5x5 convolution>
        branch3a = Conv2D(64, (1, 1), activation='relu')(x)
        branch3b = Conv2D(64, (5, 5), activation='relu')(branch3a)
        
        # Branch 4: <3x3 max pooling, 1x1 convolution>
        branch4a = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        branch4b = Conv2D(64, (1, 1), activation='relu')(branch4a)
        
        # Concatenate the outputs from all branches
        concatenated = Concatenate()([branch1, branch2b, branch3b, branch4b])
        
        return concatenated
    
    # Apply block 2 to the flattened output
    block2_output = block2(flatten)
    
    # Final fully connected layer
    output_layer = Dense(10, activation='softmax')(block2_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model