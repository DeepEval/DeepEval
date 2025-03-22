import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Path 1: Two blocks of convolution followed by average pooling
    path1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path1_conv1)
    path1_conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_pool1)
    path1_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path1_conv2)
    
    # Path 2: Single convolutional layer
    path2_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate outputs from both pathways
    concat_layer = Add()([path1_pool2, path2_conv])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model