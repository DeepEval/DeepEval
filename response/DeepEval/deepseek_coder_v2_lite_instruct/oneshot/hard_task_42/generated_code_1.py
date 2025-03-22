import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Path 1: Max pooling with 1x1 window
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        
        # Path 2: Max pooling with 2x2 window
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        
        # Path 3: Max pooling with 4x4 window
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        
        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        # Regularize with dropout
        output_tensor = Dropout(0.5)(output_tensor)
        
        return output_tensor
    
    block1_output = block1(input_tensor=input_layer)
    
    # Fully connected layer and reshape to prepare for block 2
    dense_layer = Dense(units=256, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 16))(dense_layer)  # 4x4x16 tensor
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 followed by 1x7 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 followed by alternating 7x1 and 1x7 convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: 1x1 convolution followed by average pooling
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path4)
        
        # Concatenate the outputs of the four paths along the channel dimension
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return output_tensor
    
    block2_output = block2(input_tensor=reshape_layer)
    
    # Flatten the output of block 2
    flatten_layer = Flatten()(block2_output)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model