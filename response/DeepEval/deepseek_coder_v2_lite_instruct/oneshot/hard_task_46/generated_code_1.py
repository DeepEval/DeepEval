import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(x):
        # Split the input into three groups along the channel axis
        splits = tf.split(x, 3, axis=-1)
        
        # Apply separable convolutions on each group
        conv1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(splits[0])
        conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(splits[1])
        conv5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(splits[2])
        
        # Concatenate the outputs
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])
        
        return concatenated
    
    first_block_output = first_block(input_layer)
    batch_norm = BatchNormalization()(first_block_output)
    flatten_layer = Flatten()(batch_norm)

    # Second block
    def second_block(x):
        # Branch 1: 3x3 convolution
        branch1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        
        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        branch2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
        branch2 = Conv2D(128, (3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(128, (3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max pooling
        branch3 = MaxPooling2D((3, 3), strides=1, padding='same')(x)
        
        # Concatenate the outputs
        concatenated = Concatenate()([branch1, branch2, branch3])
        
        return concatenated
    
    second_block_output = second_block(flatten_layer)
    global_avg_pool = GlobalAveragePooling2D()(second_block_output)
    dense_layer = Dense(units=10, activation='softmax')(global_avg_pool)

    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)

    return model