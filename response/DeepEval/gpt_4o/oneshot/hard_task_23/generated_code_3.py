import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 Convolutional Layer
    initial_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: Local Feature Extraction
    branch1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    branch1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1_conv1)
    
    # Branch 2: Downsampling and Upsampling
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch2_conv)
    
    # Branch 3: Downsampling and Upsampling
    branch3_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3_conv)
    
    # Concatenate the branches
    concat = Concatenate()([branch1_conv2, branch2_upsample, branch3_upsample])
    
    # Refine with a 1x1 Convolutional Layer
    refined_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Flatten and Fully Connected Layer for Classification
    flatten_layer = Flatten()(refined_conv)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model