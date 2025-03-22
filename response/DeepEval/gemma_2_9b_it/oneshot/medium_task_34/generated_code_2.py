import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    drop = Dropout(rate=0.25)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)

    # Part 3: Upsampling and Reconstruction
    up1 = UpSampling2D(size=(2, 2))(conv5)
    merge1 = Concatenate()([up1, conv3])
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge1)
    
    up2 = UpSampling2D(size=(2, 2))(conv6)
    merge2 = Concatenate()([up2, conv2])
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    merge3 = Concatenate()([up3, conv1])
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge3)

    # Final Output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model