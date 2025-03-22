import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras import regularizers
from keras.regularizers import l2

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add maxpooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    
    # Step 4: Define a block
    def block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: Average pooling followed by 1x1 convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        
        # Concatenate the outputs of these paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    # Step 5: Add block
    block_output = block(conv)
    
    # Step 6: Add batch normalization layer
    bath_norm = BatchNormalization()(block_output)
    
    # Step 7: Add flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # Step 8: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 9: Add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 10: Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model