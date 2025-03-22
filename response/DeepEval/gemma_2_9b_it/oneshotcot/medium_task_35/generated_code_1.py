import keras
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, UpSampling2D, BatchNormalization, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Downsampling
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Stage 2: Feature Extraction
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    drop3 = Dropout(0.25)(conv3)
    
    # Upsampling and Reconstruction
    up4 = UpSampling2D(size=(2, 2))(drop3)
    merge4 = concatenate([up4, conv2], axis=3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = concatenate([up5, conv1], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    # Output Layer
    output_layer = Conv2D(10, (1, 1), activation='softmax')(conv5) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model