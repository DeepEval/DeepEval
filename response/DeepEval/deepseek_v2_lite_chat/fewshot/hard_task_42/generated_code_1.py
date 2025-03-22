import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Path 1: 1x1 Max Pooling
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        # Path 2: 2x2 Max Pooling
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        # Path 3: 4x4 Max Pooling
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        
        # Flatten and dropout for regularization
        flatten = Flatten()(maxpool3)
        dropout = Dropout(0.5)(flatten)
        
        # Concatenate all paths
        output_tensor = Concatenate()([dropout, dropout, dropout, dropout])
        return output_tensor
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 convolution, 1x7 convolution, 7x1 convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: Average Pooling, 1x1 convolution
        avgpool4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avgpool4)
        
        # Concatenate all paths
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv4, conv4])
        return output_tensor
    
    # Apply block 1 and block 2
    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)
    
    # Fully connected layer and reshape
    dense1 = Dense(units=128, activation='relu')(block2_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    reshape = Reshape(target_shape=(64,))(dense2)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(reshape)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model