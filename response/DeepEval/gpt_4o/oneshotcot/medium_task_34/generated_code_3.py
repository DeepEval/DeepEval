import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction part: three pairs of (Conv, MaxPooling)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Enhancement part: (Conv, Dropout, Conv)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    dropout = Dropout(rate=0.5)(conv4)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(dropout)
    
    # Upsampling part with skip connections
    upconv1 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv5)
    concat1 = Concatenate()([upconv1, conv3])
    
    upconv2 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(concat1)
    concat2 = Concatenate()([upconv2, conv2])
    
    upconv3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(concat2)
    concat3 = Concatenate()([upconv3, conv1])
    
    # Final 1x1 Conv layer for class probabilities
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax', padding='same')(concat3)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model