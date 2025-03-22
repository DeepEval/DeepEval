import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Part 1: Feature extraction with convolutional and max-pooling layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    
    # Part 2: Enhance generalization with additional convolutional and dropout layers
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(dropout)
    
    # Part 3: Upsampling with transposed convolutions and skip connections
    up_conv1 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv5)
    upsample1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(up_conv1)
    skip1 = Add()([upsample1, conv3])  # Skip connection
    
    up_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(skip1)
    upsample2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(up_conv2)
    skip2 = Add()([upsample2, conv2])  # Skip connection
    
    up_conv3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(skip2)
    upsample3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(up_conv3)
    skip3 = Add()([upsample3, conv1])  # Skip connection
    
    # Final 1x1 convolution to generate class probabilities
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(skip3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model