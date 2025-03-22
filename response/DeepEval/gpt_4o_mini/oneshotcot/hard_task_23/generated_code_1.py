import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have size 32x32x3
    
    # Initial 1x1 Convolutional Layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: Local Feature Extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
    
    # Branch 2: Average Pooling followed by Convolution and Upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2, padding='same')(branch2)  # Upsampling
    
    # Branch 3: Average Pooling followed by Convolution and Upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=2, padding='same')(branch3)  # Upsampling
    
    # Concatenating all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 Convolutional Layer
    final_conv = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(concatenated)
    
    # Flatten and Dense Layer
    flatten_layer = Flatten()(final_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model