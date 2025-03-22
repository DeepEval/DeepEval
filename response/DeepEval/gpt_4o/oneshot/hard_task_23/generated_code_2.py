import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 Convolutional Layer
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: Local Feature Extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    
    # Branch 2: Downsampling, Convolution, Upsampling
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2)
    
    # Branch 3: Downsampling, Convolution, Upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Refinement using a 1x1 Convolutional Layer
    refined_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    
    # Flatten and Dense Layers
    flatten_layer = Flatten()(refined_output)
    dense_output = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_output)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model