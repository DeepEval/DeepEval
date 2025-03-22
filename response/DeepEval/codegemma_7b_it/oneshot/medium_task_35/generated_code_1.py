import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Conv2DTranspose, UpSampling2D, Add

def dl_model(): 
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Downsampling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Encoder
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    
    # Decoder
    up_sampling = UpSampling2D(size=(2, 2))(conv4)
    up_conv = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling)
    concat = Add()([up_conv, conv2])
    
    up_sampling = UpSampling2D(size=(2, 2))(concat)
    up_conv = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling)
    concat = Add()([up_conv, conv1])
    
    conv_output = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(concat)
    
    model = keras.Model(inputs=input_layer, outputs=conv_output)
    
    return model