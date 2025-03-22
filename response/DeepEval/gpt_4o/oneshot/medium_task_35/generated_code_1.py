import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First stage of convolution and max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Second stage of convolution and max pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    drop1 = Dropout(0.5)(conv3)

    # Upsampling with skip connections
    up1 = UpSampling2D(size=(2, 2))(drop1)
    concat1 = Concatenate()([up1, conv2])  # Skip connection from conv2
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat1)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = Concatenate()([up2, conv1])  # Skip connection from conv1
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat2)
    
    # Final output layer with 1x1 convolution
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(conv5)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model