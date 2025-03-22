import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 image size

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two parallel 1x3 and 3x1 convolutions and concatenation
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    path3_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path3)
    path3_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path3)
    path3 = Concatenate()([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by a 3x3 convolution, then followed by two parallel 1x3 and 3x1 convolutions and concatenation
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    path4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(path4)
    path4_1x3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path4)
    path4_3x1 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path4)
    path4 = Concatenate()([path4_1x3, path4_3x1])

    # Concatenate outputs from all paths
    multi_scale_features = Concatenate()([path1, path2, path3, path4])

    # Batch normalization and flattening
    multi_scale_features = BatchNormalization()(multi_scale_features)
    multi_scale_features = Flatten()(multi_scale_features)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(multi_scale_features)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model