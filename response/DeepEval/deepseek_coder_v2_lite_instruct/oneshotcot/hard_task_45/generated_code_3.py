import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def first_block(x):
        # Split the input into three groups
        splits = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=3, axis=-1))(x)
        
        # Apply depthwise separable convolutions with different kernel sizes
        conv1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(splits[0])
        conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(splits[1])
        conv5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(splits[2])
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        return concatenated
    
    first_block_output = first_block(input_layer)
    batch_normalized = BatchNormalization()(first_block_output)
    
    # Second block
    def second_block(x):
        # First branch
        branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch1)
        
        # Second branch
        branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
        
        # Third branch
        branch3 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
        
        # Fourth branch
        branch4 = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        branch4 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch4)
        
        # Fifth branch
        branch5 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        branch5 = MaxPooling2D((3, 3), strides=1, padding='same')(branch5)
        branch5 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch5)
        
        # Concatenate all branches
        concatenated = Concatenate()([branch1, branch2, branch3, branch4, branch5])
        return concatenated
    
    second_block_output = second_block(batch_normalized)
    flattened = Flatten()(second_block_output)
    dense_layer = Dense(128, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model