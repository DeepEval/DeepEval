import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Conv2DTranspose

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    # First part: feature extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Second part: enhance generalization capabilities
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)
    dropout = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout)

    # Third part: upsampling and restore spatial information
    def upsampling_block(input_tensor):
        conv = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        skip_connection = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)
        output_tensor = Concatenate()([conv, skip_connection])
        
        return output_tensor
    
    upsampling_output1 = upsampling_block(conv5)
    upsampling_output2 = upsampling_block(upsampling_output1)
    upsampling_output3 = upsampling_block(upsampling_output2)

    # Final part: probability output for 10 classes
    conv6 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(upsampling_output3)

    model = keras.Model(inputs=input_layer, outputs=conv6)

    return model