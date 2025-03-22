import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Concatenate, BatchNormalization, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block - Feature Extraction
    def block_1(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutional layers
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        
        # Batch normalization
        bn1 = BatchNormalization()(conv1)
        bn2 = BatchNormalization()(conv2)
        bn3 = BatchNormalization()(conv3)
        
        # Concatenate outputs from all three groups
        concat = Concatenate(axis=-1)([bn1, bn2, bn3])
        
        return concat
    
    # Second Block - Additional Feature Extraction
    def block_2(input_tensor):
        # Three branches for feature extraction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Average pooling
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        
        # Concatenate outputs from all branches
        concat = Concatenate()([branch1, branch2, branch3, avg_pool])
        
        return concat
    
    # Connect the blocks
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    
    # Flatten and fully connected layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model