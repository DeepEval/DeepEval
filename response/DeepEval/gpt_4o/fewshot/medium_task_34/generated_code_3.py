import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add, UpSampling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(dropout)
    
    # Part 3: Upsampling with Skip Connections
    up1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    skip1 = Add()([up1, conv3])
    
    up2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(skip1)
    skip2 = Add()([up2, conv2])
    
    up3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(skip2)
    skip3 = Add()([up3, conv1])
    
    # Final 1x1 Convolution for Classification
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(skip3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model