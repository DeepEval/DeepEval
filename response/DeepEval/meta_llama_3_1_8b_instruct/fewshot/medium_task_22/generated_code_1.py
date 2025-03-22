import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Flatten, Dense, AveragePooling2D, DepthwiseConv2D, Reshape, Lambda
from tensorflow.keras.layers import AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutions
    branch1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch1_conv1)
    
    # Branch 2: 1x1 convolutions + 3x3 convolutions
    branch2_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_conv1)
    branch2_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch2_conv2)
    
    # Branch 3: max pooling
    branch3_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    
    # Concatenate the outputs from the three branches
    merged_layer = Concatenate()([branch1_pool1, branch2_pool1, branch3_pool1])
    
    # Convolutional layer to increase feature dimensions
    conv_layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(merged_layer)
    
    # Average pooling to downsample features
    avg_pool_layer = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_layer)
    
    # Flatten the merged layer
    flattened_layer = Flatten()(avg_pool_layer)
    
    # Fully connected layers for classification
    dense_layer1 = Dense(units=128, activation='relu')(flattened_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model