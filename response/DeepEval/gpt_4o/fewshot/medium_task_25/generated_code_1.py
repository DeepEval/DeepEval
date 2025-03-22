import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: A single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Path 3: A 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions, and concatenate the results
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3a = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3)
    path3b = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3)
    path3 = Concatenate()([path3a, path3b])
    
    # Path 4: A 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions, and concatenate the results
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4a = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4)
    path4b = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4)
    path4 = Concatenate()([path4a, path4b])
    
    # Concatenate all paths
    multi_scale_feature = Concatenate()([path1, path2, path3, path4])
    
    # Flatten and pass through a fully connected layer for classification
    flatten = Flatten()(multi_scale_feature)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model