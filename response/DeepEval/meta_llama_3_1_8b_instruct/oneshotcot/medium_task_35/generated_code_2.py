import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Stage 2
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    drop = Dropout(0.2)(conv3)
    
    # Convolutional block with upsampling using skip connections
    def block(input_tensor):
        
        path1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(input_tensor))
        path1 = Concatenate()([path1, conv2])
        path1 = BatchNormalization()(path1)
        
        path2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(input_tensor))
        path2 = Concatenate()([path2, conv1])
        path2 = BatchNormalization()(path2)
        
        output_tensor = Concatenate()([path1, path2])
        output_tensor = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
        
        return output_tensor
        
    block_output = block(max_pooling2)
    bath_norm = BatchNormalization()(block_output)
    drop2 = Dropout(0.2)(bath_norm)
    
    # Final 1x1 convolutional layer for probability outputs
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(drop2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model