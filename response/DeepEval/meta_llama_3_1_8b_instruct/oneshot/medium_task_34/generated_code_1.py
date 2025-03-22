import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Conv2DTranspose

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First part: feature extraction through 3 pairs of convolutional and max-pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    
    # Second part: generalization capability enhancement through convolutional and dropout layers
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling3)
    dropout = Dropout(rate=0.2)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(dropout)
    
    # Third part: upsampling through 3 pairs of convolutional and transposed convolutional layers with skip connections
    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(up1)
    merge1 = Concatenate()([conv2, conv6])
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(merge1)
    
    up2 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(up2)
    merge2 = Concatenate()([conv1, conv8])
    conv9 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(merge2)
    
    up3 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(up3)
    
    # Final 1x1 convolutional layer for probability output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv10)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model