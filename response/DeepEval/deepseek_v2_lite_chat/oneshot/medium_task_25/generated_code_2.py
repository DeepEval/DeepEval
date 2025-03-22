import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, ZeroPadding2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32

    # Path 1: Single 1x1 convolution
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)

    # Path 3: 1x1 convolution followed by two parallel convolutions
    path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3_1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='valid', activation='relu')(path3)
    path3_3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='valid', activation='relu')(path3)
    path3 = Concatenate(axis=-1)([path3, path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by a 3x3 convolution
    path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path4)

    # Combine outputs of the parallel paths
    output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])

    # Batch normalization and flattening
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = Flatten()(output_tensor)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()