import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2_pool)

    # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
    path3_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path3_conv1x1)
    path3_conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path3_conv1x1)
    path3 = Concatenate()([path3_conv1x3, path3_conv3x1])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
    path4_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4_conv1x1)
    path4_conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(path4_conv3x3)
    path4_conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path4_conv3x3)
    path4 = Concatenate()([path4_conv1x3, path4_conv3x1])

    # Concatenate all paths to form a multi-scale feature fusion
    multi_scale_feature_fusion = Concatenate()([path1, path2, path3, path4])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(multi_scale_feature_fusion)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model