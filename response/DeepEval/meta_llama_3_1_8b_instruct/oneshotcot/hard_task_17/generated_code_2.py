import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Reshape, Permute, GlobalAveragePooling2D

def dl_model():
    
    # Construct Block 1
    input_layer = Input(shape=(32, 32, 3))
    block1_input = input_layer
    
    # Extract deep features using two 3x3 convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_input)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Apply global average pooling to generate weights
    weights = GlobalAveragePooling2D()(conv2)
    
    # Reshape and multiply with the input to produce the weighted feature output
    reshaped_weights = Reshape((1, 1, 32))(weights)
    weighted_output = Multiply()([input_layer, reshaped_weights])
    
    # Construct Block 2
    block2_input = weighted_output
    
    # Extract deep features using two 3x3 convolutional layers followed by a max pooling layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_input)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv4)
    
    # Connect the branch from Block 1 directly to the output of Block 2
    branch = block1_input
    
    # Fuse the outputs from the main path and the branch
    combined_output = Add()([max_pooling, branch])
    
    # Apply two fully connected layers to classify the combined output
    dense1 = Dense(units=64, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model