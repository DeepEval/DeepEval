import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Stage 2: Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Additional Convolution and Dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    drop1 = Dropout(rate=0.3)(conv3)
    
    # Upsampling and skip connections
    up1 = UpSampling2D(size=(2, 2))(drop1)
    skip1 = Concatenate()([up1, conv2])
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Concatenate()([up2, conv1])
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    
    # Final 1x1 Convolution for 10-category output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(conv5)
    
    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model