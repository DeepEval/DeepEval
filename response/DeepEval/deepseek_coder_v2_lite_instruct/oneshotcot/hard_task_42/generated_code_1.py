import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Path 1: 1x1 MaxPooling
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(0.25)(path1)
        
        # Path 2: 2x2 MaxPooling
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(0.25)(path2)
        
        # Path 3: 4x4 MaxPooling
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(0.25)(path3)
        
        # Concatenate outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Fully connected layer and reshape for Block 2
    reshape_layer = Dense(4, activation='relu')(block1_output)
    reshape_layer = Reshape((2, 2, 4))(reshape_layer)
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Conv2D
        path1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 -> 1x7 -> 7x1 Conv2D
        path2 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(64, kernel_size=(1, 7), padding='same', activation='relu')(path2)
        path2 = Conv2D(64, kernel_size=(7, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 -> 7x1 -> 1x7 Conv2D
        path3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(64, kernel_size=(7, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(64, kernel_size=(1, 7), padding='same', activation='relu')(path3)
        
        # Path 4: Average pooling followed by 1x1 Conv2D
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(input_tensor)
        path4 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        
        # Concatenate outputs of the four paths along the channel dimension
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        return output_tensor
    
    block2_output = block2(reshape_layer)
    
    # Flatten the output of Block 2
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model