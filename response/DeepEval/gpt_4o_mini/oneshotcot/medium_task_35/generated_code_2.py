import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)
    
    # Stage 1: Convolution and Max Pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Stage 2: Convolution and Max Pooling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Additional Convolutions and Dropout
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    drop1 = Dropout(0.5)(conv3)

    # Upsampling and Skip Connections
    upsample1 = UpSampling2D(size=(2, 2))(drop1)
    skip1 = Concatenate()([upsample1, conv2])  # Skip connection
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(skip1)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv4)

    upsample2 = UpSampling2D(size=(2, 2))(conv4)
    skip2 = Concatenate()([upsample2, conv1])  # Skip connection
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(skip2)
    conv5 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv5)

    # Final 1x1 Convolutional Layer for Output
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(conv5)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model