import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, AveragePooling2D

def dl_model():
    
    # Define the initial convolutional layer to reduce dimensionality
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the basic block with a convolutional layer, batch normalization, and ReLU activation
    def basic_block(input_tensor):
        conv_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv_path)
        branch_path = input_tensor
        adding_layer = Add()([batch_norm, branch_path])
        return adding_layer
    
    # Define the main structure of the model with two consecutive basic blocks
    block1 = basic_block(conv1)
    block2 = basic_block(block1)
    
    # Define the branch with another convolutional layer
    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both paths
    adding_layer = Add()([block2, branch])
    
    # Apply average pooling to downsample the feature map
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(adding_layer)
    
    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model