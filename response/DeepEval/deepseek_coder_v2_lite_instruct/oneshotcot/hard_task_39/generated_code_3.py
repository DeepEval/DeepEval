import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    
    # MaxPooling layers with different pool sizes and strides
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the results of each pooling layer
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened results
    concatenated_features = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer after Block 1
    fc1 = Dense(units=128, activation='relu')(concatenated_features)
    
    # Reshape the output to be suitable for Block 2
    reshaped_output = Reshape((1, 1, 128))(fc1)
    
    # Block 2
    # Multiple branches for different convolutions and pooling
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)
    
    # Concatenate the outputs of all branches
    concatenated_features_block2 = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Flatten the concatenated features
    flattened_output = Flatten()(concatenated_features_block2)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model