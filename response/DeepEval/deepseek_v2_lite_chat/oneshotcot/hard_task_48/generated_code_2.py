import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Block 1
    def block1(input_tensor):
        split = Lambda(lambda x: tf.split(input_tensor, num_split=3, axis=1))(input_tensor)
        
        # Paths for block 1
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(split[0])
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split[1])
        path3 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split[2])
        
        # Concatenate the outputs of paths 1, 2, and 3
        concat_path123 = Concatenate(axis=1)([path1, path2, path3])
        
        # Batch normalization and flattening for block 1
        batch_norm_block1 = BatchNormalization()(concat_path123)
        flat_block1 = Flatten()(batch_norm_block1)
        
        return flat_block1
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        
        # Path 2: 3x3 average pooling, 1x1 convolution
        path2 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(path2)
        
        # Path 3: 1x1 convolution, two sub-paths
        path3_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu', padding='same')(path3_1x1)
        concat_path3 = Concatenate(axis=1, name='path3_concat')([path3_3x1, path3_1x1])
        
        # Path 4: 1x1 convolution, 3x3 convolution, two sub-paths
        path4_1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path4_3x3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(path4_1x1)
        path4_3x1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(path4_3x3)
        concat_path4 = Concatenate(axis=1, name='path4_concat')([path4_3x1_1, path4_3x3])
        
        # Concatenate the outputs of paths 1, 2, 3, and 4
        concat_block2 = Concatenate(axis=1)([path1, path2, concat_path3, concat_path4])
        
        # Batch normalization and flattening for block 2
        batch_norm_block2 = BatchNormalization()(concat_block2)
        flat_block2 = Flatten()(batch_norm_block2)
        
        return flat_block2
    
    # Build the model
    input_layer = input_layer
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model