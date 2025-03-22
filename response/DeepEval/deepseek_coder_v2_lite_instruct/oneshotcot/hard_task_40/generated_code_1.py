import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dropout
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add average pooling layers
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(conv)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv)
    pool4 = AveragePooling2D(pool_size=(4, 4), strides=4)(conv)
    
    # Step 4: Concatenate the outputs from the pooling layers
    concat_layer = Concatenate(axis=-1)([Flatten()(pool1), Flatten()(pool2), Flatten()(pool4)])
    
    # Step 5: Reshape the concatenated output into a 4-dimensional tensor
    reshape_layer = Reshape((4, 1))(concat_layer)
    
    # Step 6: Define the second block
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 followed by two 3x3 convolutions
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 followed by a single 3x3 convolution
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: 1x1 convolution followed by average pooling
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        
        # Dropout for regularization
        path1 = Dropout(0.25)(path1)
        path2 = Dropout(0.25)(path2)
        path3 = Dropout(0.25)(path3)
        path4 = Dropout(0.25)(path4)
        
        # Concatenate the outputs from all paths
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return output_tensor
    
    # Apply the second block
    block_output = second_block(reshape_layer)
    
    # Step 7: Add batch normalization
    batch_norm = BatchNormalization()(block_output)
    
    # Step 8: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 9: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 10: Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 11: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Step 12: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()