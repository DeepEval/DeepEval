import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution + MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Stage 2: Convolution + MaxPooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    drop1 = Dropout(0.5)(conv3)
    
    # Upsampling with skip connections
    up1 = UpSampling2D(size=(2, 2))(drop1)
    skip1 = Add()([up1, conv2])  # Skip connection from conv2
    
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Add()([up2, conv1])  # Skip connection from conv1
    
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    
    # Final 1x1 convolutional layer for classification into 10 categories
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(conv5)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model