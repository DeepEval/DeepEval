import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: Local features extraction through 3x3 convolution
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Branch 2: Max pooling, 3x3 convolution, and upsampling
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    
    # Branch 3: Max pooling, 3x3 convolution, and upsampling
    pool3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    
    # Upsampling and merging branches
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4)
    merge2 = Concatenate()([upconv3, conv3])
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(merge2)
    
    # Upsampling and merging branches
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(upconv2)
    merge3 = Concatenate()([upconv2, conv2])
    upconv1 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(merge3)
    
    # Final 1x1 convolution for classification
    conv5 = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(upconv1)
    
    model = keras.Model(inputs=input_layer, outputs=conv5)
    return model