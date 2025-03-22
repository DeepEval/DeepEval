import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))

    # Part 1: Feature Extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Part 2: Generalization Enhancement
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(max_pool3)
    dropout = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(dropout)

    # Part 3: Upsampling and Reconstruction
    upconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(conv5)
    concat1 = Concatenate()([upconv1, conv2])

    upconv2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(concat1)
    concat2 = Concatenate()([upconv2, conv1])

    output_conv = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(concat2)

    model = keras.Model(inputs=input_layer, outputs=output_conv)

    return model