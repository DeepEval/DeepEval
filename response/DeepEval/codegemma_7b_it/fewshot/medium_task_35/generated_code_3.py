import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Dropout, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Downsampling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Feature Processing
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    drop4 = Dropout(rate=0.2)(conv4)
    
    # Upsampling
    up5 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(drop4)
    up5 = UpSampling2D(size=(2, 2))(up5)
    merge5 = Add()([up5, conv2])
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge5)
    
    up6 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    up6 = UpSampling2D(size=(2, 2))(up6)
    merge6 = Add()([up6, conv1])
    conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merge6)
    
    # Output
    conv7 = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv6)
    
    model = keras.Model(inputs=input_layer, outputs=conv7)
    
    return model